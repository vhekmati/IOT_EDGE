#!/usr/bin/env python3

import argparse
import base64
import csv
import json
import os
import secrets
import ssl
import threading
import time
from pathlib import Path

import paho.mqtt.client as mqtt
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


class ProducerRunner:
    def __init__(self, local_config_path):
        with open(local_config_path, "r", encoding="utf-8") as input_file:
            self.local = json.load(input_file)

        self.broker = self.local["broker"]
        self.topics = self.local["topics"]
        self.key = bytes.fromhex(self.local["key_hex"])
        self.output_root = Path(self.local["output_root"])
        self.output_root.mkdir(parents=True, exist_ok=True)

        self.window = int(self.local.get("window", 120))
        self.step = int(self.local.get("step", 1))
        self.sensor_numbers = self.local.get(
            "sensor_numbers",
            [2, 3, 4, 7, 11, 12, 15],
        )

        rows = self._load_cmapss_rows(self.local["data_path"])
        self.windows = self._build_windows(rows)

        self.client = mqtt.Client(
            client_id=self.local.get(
                "client_id",
                "benchmark-producer-runner",
            ),
            clean_session=True,
            protocol=mqtt.MQTTv311,
        )
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.tls_set(
            ca_certs=self.broker["ca_certs"],
            certfile=self.broker["certfile"],
            keyfile=self.broker["keyfile"],
            cert_reqs=ssl.CERT_REQUIRED,
        )

        self.busy_lock = threading.Lock()
        self.busy = False
        self.current_experiment_id = None

    @staticmethod
    def _sensor_numbers_to_columns(sensor_numbers):
        return [5 + (sensor_number - 1) for sensor_number in sensor_numbers]

    @staticmethod
    def _load_cmapss_rows(path):
        rows = []
        with open(path, "r", encoding="utf-8") as input_file:
            for line in input_file:
                values = line.strip().split()
                if values:
                    rows.append([float(value) for value in values])
        return rows

    def _build_windows(self, rows):
        sensor_columns = self._sensor_numbers_to_columns(
            self.sensor_numbers
        )
        engines = {}

        for row in rows:
            unit_id = int(row[0])
            engines.setdefault(unit_id, [])
            engines[unit_id].append([
                row[column] for column in sensor_columns
            ])

        windows = []
        for unit_id in sorted(engines):
            engine_data = engines[unit_id]
            if len(engine_data) < self.window:
                continue
            for start in range(
                0,
                len(engine_data) - self.window + 1,
                self.step,
            ):
                windows.append(
                    engine_data[start : start + self.window]
                )
        return windows

    @staticmethod
    def _pad(data):
        padding_length = 16 - (len(data) % 16)
        return data + bytes([padding_length] * padding_length)

    def _encrypt_payload(self, plaintext):
        iv = secrets.token_bytes(16)
        padded = self._pad(plaintext.encode("utf-8"))
        cipher = Cipher(
            algorithms.AES(self.key),
            modes.CBC(iv),
            backend=default_backend(),
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(padded) + encryptor.finalize()

        return json.dumps(
            {
                "v": 1,
                "alg": "AES-256-CBC",
                "iv": base64.b64encode(iv).decode("ascii"),
                "ct": base64.b64encode(ciphertext).decode("ascii"),
            },
            separators=(",", ":"),
        )

    def _publish_status(self, payload, wait=False):
        encoded = json.dumps(payload, separators=(",", ":"))
        info = self.client.publish(
            self.topics["producer_status"],
            encoded,
            qos=1,
            retain=False,
        )
        if wait:
            info.wait_for_publish()

    def _on_connect(self, client, userdata, flags, rc):
        del userdata, flags
        if rc == 0:
            client.subscribe(
                self.topics["producer_control"],
                qos=1,
            )
            print("PRODUCER_RUNNER_READY", flush=True)
        else:
            print(f"PRODUCER_CONNECT_FAILED rc={rc}", flush=True)

    def _on_message(self, client, userdata, msg):
        del client, userdata
        try:
            command = json.loads(msg.payload.decode("utf-8"))
        except Exception:
            return

        action = command.get("command")
        if action == "PING":
            self._publish_status({
                "status": "PONG",
                "request_id": command.get("request_id", ""),
            })
            return

        if action != "START":
            return

        experiment_id = command.get("experiment_id", "")
        with self.busy_lock:
            if self.busy:
                if self.current_experiment_id == experiment_id:
                    self._publish_status({
                        "status": "STARTED",
                        "experiment_id": experiment_id,
                        "duplicate": True,
                    })
                else:
                    self._publish_status({
                        "status": "ERROR",
                        "experiment_id": experiment_id,
                        "error": "Producer runner is busy",
                    })
                return

            self.busy = True
            self.current_experiment_id = experiment_id

        worker = threading.Thread(
            target=self._run_experiment,
            args=(command,),
            daemon=True,
        )
        worker.start()

    @staticmethod
    def _atomic_write_csv(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        with temporary_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as output_file:
            writer = csv.writer(output_file)
            writer.writerow([
                "experiment_id",
                "seq_id",
                "t1_producer_send_ns",
            ])
            writer.writerows(rows)
        os.replace(temporary_path, path)

    def _run_experiment(self, config):
        experiment_id = config["experiment_id"]
        total_messages = int(config["total_messages"])
        interval = float(config["publish_interval_s"])
        qos = int(config.get("mqtt_qos", 0))
        wait_for_publish = bool(config.get("wait_for_publish", False))
        mode = config["mode"]

        try:
            if len(self.windows) < total_messages:
                raise RuntimeError(
                    f"Not enough windows: {len(self.windows)} < {total_messages}"
                )

            # Build and encrypt every message before the timed publishing
            # phase. Preparation time must not be added to the configured
            # interval between consecutive MQTT publishes.
            prepared_messages = []

            for index in range(total_messages):
                seq_id = index + 1
                window = self.windows[index]
                payload = {
                    "experiment_id": experiment_id,
                    "seq_id": seq_id,
                    "window_shape": [len(window), len(window[0])],
                    "data": window,
                }
                encrypted = self._encrypt_payload(
                    json.dumps(payload, separators=(",", ":"))
                )
                prepared_messages.append((seq_id, encrypted))

            self._publish_status({
                "status": "STARTED",
                "experiment_id": experiment_id,
                "start_ns": time.time_ns(),
            }, wait=True)

            timestamp_rows = []
            producer_finish_ns = None
            schedule_start = time.perf_counter()

            for index, prepared in enumerate(prepared_messages):
                seq_id, encrypted = prepared

                # Use absolute deadlines. Therefore publish_interval_s means
                # T1(n+1) - T1(n), instead of an extra sleep after a send.
                if index > 0 and interval > 0:
                    deadline = schedule_start + (index * interval)
                    remaining = deadline - time.perf_counter()
                    if remaining > 0:
                        time.sleep(remaining)

                t1_ns = (
                    time.time_ns()
                    if mode == "deployment"
                    else None
                )

                info = self.client.publish(
                    self.topics["data"],
                    encrypted,
                    qos=qos,
                )

                if mode == "deployment":
                    timestamp_rows.append([
                        experiment_id,
                        seq_id,
                        t1_ns,
                    ])

                if wait_for_publish:
                    info.wait_for_publish()

            producer_finish_ns = time.time_ns()

            output_path = ""
            if mode == "deployment":
                output_path = (
                    self.output_root
                    / experiment_id
                    / "producer_timestamps.csv"
                )
                self._atomic_write_csv(output_path, timestamp_rows)

            done_status = {
                "status": "DONE",
                "experiment_id": experiment_id,
                "sent_messages": total_messages,
                "producer_finish_ns": producer_finish_ns,
                "status_publish_ns": time.time_ns(),
                "output_path": str(output_path),
            }
            self._publish_status(done_status, wait=True)

        except Exception as error:
            self._publish_status({
                "status": "ERROR",
                "experiment_id": experiment_id,
                "error": repr(error),
            }, wait=True)
        finally:
            with self.busy_lock:
                self.busy = False
                self.current_experiment_id = None

    def run(self):
        self.client.connect(
            self.broker["host"],
            self.broker["port"],
        )
        self.client.loop_forever()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).resolve().parent / "producer_local_config.json"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ProducerRunner(args.config).run()


if __name__ == "__main__":
    main()
