#!/usr/bin/env python3

import argparse
import base64
import csv
import json
import os
import ssl
import threading
import time
from pathlib import Path

import paho.mqtt.client as mqtt
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


class CloudClientRunner:
    def __init__(self, local_config_path):
        with open(local_config_path, "r", encoding="utf-8") as input_file:
            self.local = json.load(input_file)

        self.broker = self.local["broker"]
        self.topics = self.local["topics"]
        self.key = bytes.fromhex(self.local["key_hex"])
        self.output_root = Path(self.local["output_root"])
        self.output_root.mkdir(parents=True, exist_ok=True)

        self.client = mqtt.Client(
            client_id=self.local.get(
                "client_id",
                "benchmark-cloud-runner",
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
            tls_version=ssl.PROTOCOL_TLS_CLIENT,
        )

        self.state_lock = threading.Lock()
        self.active_experiment_id = None
        self.active_mode = None
        self.expected_messages = 0
        self.received_count = 0
        self.last_arrival_ns = None
        self.timestamp_rows = []
        self.complete_sent = False

    def _decrypt_payload(self, encrypted_message):
        envelope = json.loads(encrypted_message)
        iv = base64.b64decode(envelope["iv"])
        ciphertext = base64.b64decode(envelope["ct"])

        cipher = Cipher(
            algorithms.AES(self.key),
            modes.CBC(iv),
            backend=default_backend(),
        )
        decryptor = cipher.decryptor()
        padded = decryptor.update(ciphertext) + decryptor.finalize()

        unpadder = padding.PKCS7(128).unpadder()
        plaintext = unpadder.update(padded) + unpadder.finalize()
        return json.loads(plaintext.decode("utf-8"))

    def _publish_status(self, payload, wait=False):
        encoded = json.dumps(payload, separators=(",", ":"))
        info = self.client.publish(
            self.topics["consumer_status"],
            encoded,
            qos=1,
            retain=False,
        )
        if wait:
            info.wait_for_publish()

    def _on_connect(self, client, userdata, flags, rc):
        del userdata, flags
        if rc == 0:
            client.subscribe(self.topics["data"], qos=1)
            client.subscribe(
                self.topics["consumer_control"],
                qos=1,
            )
            print("CLOUD_RUNNER_READY", flush=True)
        else:
            print(f"CLOUD_CONNECT_FAILED rc={rc}", flush=True)

    def _on_message(self, client, userdata, msg):
        t4_ns = time.time_ns()
        del client, userdata

        if msg.topic == self.topics["consumer_control"]:
            self._handle_control(msg)
            return

        if msg.topic != self.topics["data"]:
            return

        with self.state_lock:
            active_experiment_id = self.active_experiment_id
            active_mode = self.active_mode

        if active_experiment_id is None:
            return

        try:
            payload = self._decrypt_payload(
                msg.payload.decode("utf-8")
            )
        except Exception:
            return

        if payload.get("experiment_id") != active_experiment_id:
            return

        publish_complete = False
        complete_payload = None

        with self.state_lock:
            if self.active_experiment_id != active_experiment_id:
                return

            self.received_count += 1
            self.last_arrival_ns = t4_ns

            if active_mode == "deployment":
                self.timestamp_rows.append([
                    active_experiment_id,
                    payload.get("seq_id", ""),
                    t4_ns,
                ])

            if (
                self.expected_messages > 0
                and self.received_count >= self.expected_messages
                and not self.complete_sent
            ):
                self.complete_sent = True
                publish_complete = True
                complete_payload = {
                    "status": "COMPLETE",
                    "experiment_id": active_experiment_id,
                    "received_messages": self.received_count,
                    "last_arrival_ns": self.last_arrival_ns,
                }

        if publish_complete:
            self._publish_status(complete_payload)

    def _handle_control(self, msg):
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

        experiment_id = command.get("experiment_id", "")

        if action == "START":
            error = None
            duplicate = False
            with self.state_lock:
                if self.active_experiment_id is None:
                    self.active_experiment_id = experiment_id
                    self.active_mode = command.get("mode", "deployment")
                    self.expected_messages = int(
                        command.get("expected_messages", 0)
                    )
                    self.received_count = 0
                    self.last_arrival_ns = None
                    self.timestamp_rows = []
                    self.complete_sent = False
                elif self.active_experiment_id == experiment_id:
                    duplicate = True
                else:
                    error = (
                        "Consumer runner is already active for "
                        + str(self.active_experiment_id)
                    )

            if error is not None:
                self._publish_status({
                    "status": "ERROR",
                    "experiment_id": experiment_id,
                    "error": error,
                })
            else:
                self._publish_status({
                    "status": "READY",
                    "experiment_id": experiment_id,
                    "ready_ns": time.time_ns(),
                    "duplicate": duplicate,
                })
            return

        if action == "STATUS":
            with self.state_lock:
                response = {
                    "status": "STATUS",
                    "request_id": command.get("request_id", ""),
                    "experiment_id": experiment_id,
                    "active_experiment_id": self.active_experiment_id,
                    "received_messages": self.received_count,
                    "expected_messages": self.expected_messages,
                    "last_arrival_ns": self.last_arrival_ns,
                }
            self._publish_status(response)
            return

        if action == "STOP":
            with self.state_lock:
                if self.active_experiment_id != experiment_id:
                    self._publish_status({
                        "status": "STOPPED",
                        "experiment_id": experiment_id,
                        "received_messages": 0,
                        "stop_ns": time.time_ns(),
                        "output_path": "",
                        "ignored": True,
                    })
                    return

                rows = list(self.timestamp_rows)
                mode = self.active_mode
                received_count = self.received_count
                last_arrival_ns = self.last_arrival_ns
                self.active_experiment_id = None
                self.active_mode = None
                self.expected_messages = 0
                self.received_count = 0
                self.last_arrival_ns = None
                self.timestamp_rows = []
                self.complete_sent = False

            output_path = ""
            if mode == "deployment":
                output_path = (
                    self.output_root
                    / experiment_id
                    / "consumer_timestamps.csv"
                )
                self._atomic_write_csv(output_path, rows)

            self._publish_status({
                "status": "STOPPED",
                "experiment_id": experiment_id,
                "received_messages": received_count,
                "last_arrival_ns": last_arrival_ns,
                "stop_ns": time.time_ns(),
                "output_path": str(output_path),
            })

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
                "t4_consumer_arrival_ns",
            ])
            writer.writerows(rows)
        os.replace(temporary_path, path)

    def run(self):
        self.client.connect(
            self.broker["host"],
            self.broker["port"],
            keepalive=60,
        )
        self.client.loop_forever()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).resolve().parent / "cloud_local_config.json"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    CloudClientRunner(args.config).run()


if __name__ == "__main__":
    main()
