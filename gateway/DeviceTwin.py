import csv
import os
import ssl
import threading
import time
from pathlib import Path

import paho.mqtt.client as mqtt

from EdgeProcessing import EdgeProcessingChain


class DeviceTwin:
    GATEWAY_HEADER = [
        "experiment_id",
        "seq_id",
        "t2_gateway_ingress_ns",
        "t3_gateway_egress_ready_ns",
    ]

    def __init__(self, config, public_client):
        self.id = config["deviceTwinId"]
        self.public_client = public_client
        self.receiver = None
        self.edge_processor = None
        self.connection_config = None
        self.experiment_config = None
        self.connected = threading.Event()
        self.processing_complete = threading.Event()
        self.gateway_rows = []

        self.processed_messages = 0
        self.results_flushed = False
        self.flush_lock = threading.Lock()

    def start_edge_processing_chain(self, chain_config):
        self.experiment_config = chain_config["experiment"]
        self.edge_processor = EdgeProcessingChain(
            processors=chain_config.get("processors"),
            device_id=self.id,
            config=self.experiment_config,
        )

    def start_connections(self, connection_config):
        self.connection_config = connection_config
        self.receiver = mqtt.Client(
            client_id=f"{self.id}-rx",
            clean_session=True,
            protocol=mqtt.MQTTv311,
        )
        self.receiver.on_connect = self._on_connect
        self.receiver.on_message = self._on_message
        self.receiver.tls_set(
            ca_certs=connection_config["receiver_ca_certs"],
            certfile=connection_config["receiver_certfile"],
            keyfile=connection_config["receiver_keyfile"],
            cert_reqs=ssl.CERT_REQUIRED,
        )
        self.receiver.connect(
            connection_config["broker_receiver_ip"],
            connection_config["broker_receiver_port"],
        )
        self.receiver.loop_start()

    def _on_connect(self, client, userdata, flags, rc):
        del userdata, flags
        if rc == 0:
            client.subscribe(
                self.connection_config["receiver_sub_topic"],
                qos=self.experiment_config.get("mqtt_qos", 0),
            )
            self.connected.set()

    def _on_message(self, client, userdata, msg):
        del client, userdata

        mode = self.experiment_config["mode"]

        if mode == "deployment":
            t2_ns = time.time_ns()

        output = self.edge_processor.process(
            msg.payload.decode("utf-8")
        )

        if output is None:
            return

        if mode == "deployment":
            t3_ns = time.time_ns()
            self.gateway_rows.append([
                self.experiment_config["experiment_id"],
                self.edge_processor.last_seq_id,
                t2_ns,
                t3_ns,
            ])

        self.public_client.publish(
            self.connection_config["cloud_pub_topic"],
            output,
            qos=self.experiment_config.get("mqtt_qos", 0),
        )

        self.processed_messages += 1

        # This is only an in-memory synchronization signal. It performs no
        # file I/O and is set after the final message has completed the full
        # processing and publish path.
        expected_messages = int(
            self.experiment_config["total_messages"]
        )
        if self.processed_messages >= expected_messages:
            self.processing_complete.set()

    @staticmethod
    def _atomic_write_csv(path, header, rows):
        path = Path(path)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        with temporary_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as output_file:
            writer = csv.writer(output_file)
            writer.writerow(header)
            writer.writerows(rows)
        os.replace(temporary_path, path)

    def flush_results(self):
        with self.flush_lock:
            if self.results_flushed:
                return

            if self.edge_processor is None:
                return

            if self.experiment_config["mode"] == "measurement":
                self.edge_processor.flush_latency()

            elif self.experiment_config["mode"] == "deployment":
                output_dir = Path(
                    self.experiment_config["output_directory"]
                )
                self._atomic_write_csv(
                    output_dir / "gateway_timestamps.csv",
                    self.GATEWAY_HEADER,
                    self.gateway_rows,
                )
                self.gateway_rows.clear()

            self.results_flushed = True

    def is_processing_complete(self):
        return self.processing_complete.is_set()

    def stop(self):
        # Results are flushed by the main thread as soon as all expected
        # messages are processed. This fallback covers early/failure shutdown.
        self.flush_results()

        # Avoid loop_stop()/backend cleanup here. Older Paho/TitanSSL builds
        # can block during teardown. Each experiment uses a dedicated process,
        # so the OS safely releases sockets, threads, and native handles when
        # the process exits.
        if self.receiver is not None:
            try:
                self.receiver.disconnect()
            except Exception:
                pass
