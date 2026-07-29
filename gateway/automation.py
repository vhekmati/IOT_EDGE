#!/usr/bin/env python3

import argparse
import json
import ssl
import threading
import time
import uuid
from pathlib import Path

import paho.mqtt.client as mqtt

from run_experiment import LocalGatewayRun


class MQTTControlEndpoint:
    def __init__(self, broker_config, client_id, status_topic):
        self.broker_config = broker_config
        self.status_topic = status_topic
        self.connected = threading.Event()
        self.condition = threading.Condition()
        self.messages = []

        self.client = mqtt.Client(
            client_id=client_id,
            clean_session=True,
            protocol=mqtt.MQTTv311,
        )
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.tls_set(
            ca_certs=broker_config["ca_certs"],
            certfile=broker_config["certfile"],
            keyfile=broker_config["keyfile"],
            cert_reqs=ssl.CERT_REQUIRED,
        )

    def _on_connect(self, client, userdata, flags, rc):
        del userdata, flags
        if rc == 0:
            client.subscribe(self.status_topic, qos=1)
            self.connected.set()

    def _on_message(self, client, userdata, msg):
        del client, userdata
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except Exception:
            return

        with self.condition:
            self.messages.append(payload)
            self.condition.notify_all()

    def start(self, timeout_s=30):
        self.client.connect(
            self.broker_config["host"],
            self.broker_config["port"],
        )
        self.client.loop_start()
        if not self.connected.wait(timeout_s):
            raise TimeoutError(
                f"MQTT connection timed out: {self.status_topic}"
            )

    def stop(self):
        self.client.loop_stop()
        self.client.disconnect()

    def publish(self, topic, payload):
        encoded = json.dumps(payload, separators=(",", ":"))
        info = self.client.publish(topic, encoded, qos=1, retain=False)
        info.wait_for_publish()

    def wait_for(self, predicate, timeout_s):
        deadline = time.time() + timeout_s
        with self.condition:
            while True:
                for index, payload in enumerate(self.messages):
                    if predicate(payload):
                        return self.messages.pop(index)

                remaining = deadline - time.time()
                if remaining <= 0:
                    raise TimeoutError("Expected status message was not received")
                self.condition.wait(remaining)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-dir",
        default="../configs/generated",
    )
    parser.add_argument(
        "--local-config",
        default="gateway_local_config.json",
    )
    parser.add_argument("--pattern", default="*.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--no-skip-completed", action="store_true")
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


def save_json_atomic(path, value):
    path = Path(path)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2)
    temporary_path.replace(path)


def wait_for_runner(
    endpoint,
    control_topic,
    role,
    total_timeout_s,
):
    deadline = time.time() + total_timeout_s

    while time.time() < deadline:
        request_id = uuid.uuid4().hex
        endpoint.publish(
            control_topic,
            {
                "command": "PING",
                "request_id": request_id,
            },
        )
        try:
            endpoint.wait_for(
                lambda item: (
                    item.get("status") == "PONG"
                    and item.get("request_id") == request_id
                ),
                timeout_s=5,
            )
            print(f"{role} runner is online", flush=True)
            return
        except TimeoutError:
            time.sleep(2)

    raise TimeoutError(f"{role} runner is not online")


def wait_experiment_status(endpoint, expected_status, experiment_id, timeout_s):
    item = endpoint.wait_for(
        lambda payload: (
            payload.get("experiment_id") == experiment_id
            and payload.get("status") in (expected_status, "ERROR")
        ),
        timeout_s=timeout_s,
    )
    if item.get("status") == "ERROR":
        raise RuntimeError(item.get("error", "Remote runner error"))
    return item


def request_consumer_status(
    endpoint,
    control_topic,
    experiment_id,
    timeout_s,
):
    request_id = uuid.uuid4().hex
    endpoint.publish(
        control_topic,
        {
            "command": "STATUS",
            "request_id": request_id,
            "experiment_id": experiment_id,
        },
    )
    return endpoint.wait_for(
        lambda payload: (
            payload.get("status") == "STATUS"
            and payload.get("request_id") == request_id
            and payload.get("experiment_id") == experiment_id
        ),
        timeout_s=timeout_s,
    )


def wait_for_consumer_drain(
    endpoint,
    control_topic,
    experiment_id,
    expected_messages,
    idle_s,
    timeout_s,
    poll_s,
):
    deadline = time.monotonic() + timeout_s
    stable_since = time.monotonic()
    previous_count = None
    last_status = None

    while time.monotonic() < deadline:
        status = request_consumer_status(
            endpoint,
            control_topic,
            experiment_id,
            timeout_s=min(10.0, timeout_s),
        )
        last_status = status
        count = int(status.get("received_messages", 0))

        if count >= expected_messages:
            return {
                "reason": "expected_count_reached",
                "received_messages": count,
                "last_arrival_ns": status.get("last_arrival_ns"),
            }

        now = time.monotonic()
        if previous_count is None or count != previous_count:
            previous_count = count
            stable_since = now
        elif now - stable_since >= idle_s:
            return {
                "reason": "idle",
                "received_messages": count,
                "last_arrival_ns": status.get("last_arrival_ns"),
            }

        time.sleep(poll_s)

    raise TimeoutError(
        "Consumer drain timed out. Last status: " + repr(last_status)
    )


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    local_config_path = Path(args.local_config)
    if not local_config_path.is_absolute():
        local_config_path = base_dir / local_config_path

    local = load_json(local_config_path)
    topics = local["topics"]

    internal_control = MQTTControlEndpoint(
        local["internal_broker"],
        "gateway-automation-internal",
        topics["producer_status"],
    )
    cloud_control = MQTTControlEndpoint(
        local["cloud_broker"],
        "gateway-automation-cloud",
        topics["consumer_status"],
    )

    internal_control.start()
    cloud_control.start()

    try:
        runner_timeout = local.get("runner_ready_timeout_s", 120)
        wait_for_runner(
            internal_control,
            topics["producer_control"],
            "Producer",
            runner_timeout,
        )
        wait_for_runner(
            cloud_control,
            topics["consumer_control"],
            "Consumer",
            runner_timeout,
        )

        config_dir = Path(args.config_dir)
        if not config_dir.is_absolute():
            config_dir = (base_dir / config_dir).resolve()

        config_paths = sorted(config_dir.glob(args.pattern))
        if args.limit is not None:
            config_paths = config_paths[: args.limit]

        if not config_paths:
            raise FileNotFoundError(
                f"No experiment configs found in {config_dir}"
            )

        for position, config_path in enumerate(config_paths, start=1):
            experiment = load_json(config_path)
            experiment_id = experiment["experiment_id"]

            output_dir = Path(experiment["output_directory"])
            if not output_dir.is_absolute():
                output_dir = base_dir / output_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = output_dir / "run_manifest.json"

            if (
                not args.no_skip_completed
                and manifest_path.exists()
                and load_json(manifest_path).get("status") == "completed"
            ):
                print(f"SKIP completed: {experiment_id}", flush=True)
                continue

            print(
                f"[{position}/{len(config_paths)}] START {experiment_id}",
                flush=True,
            )

            manifest = {
                "experiment_id": experiment_id,
                "status": "starting",
                "automation_start_ns": time.time_ns(),
                "experiment_config": experiment,
            }
            save_json_atomic(manifest_path, manifest)

            local_run = None
            try:
                local_run = LocalGatewayRun(
                    config_path,
                    local_config_path,
                )
                local_run.start()
                manifest["gateway_pid"] = local_run.gateway_pid
                manifest["gateway_ready_ns"] = time.time_ns()
                cloud_control.publish(
                    topics["consumer_control"],
                    {
                        "command": "START",
                        "experiment_id": experiment_id,
                        "mode": experiment["mode"],
                        "expected_messages": experiment["total_messages"],
                    },
                )
                consumer_ready = wait_experiment_status(
                    cloud_control,
                    "READY",
                    experiment_id,
                    local.get("status_timeout_s", 60),
                )
                manifest["consumer_ready"] = consumer_ready

                producer_command = dict(experiment)
                producer_command["command"] = "START"
                manifest["producer_start_command_ns"] = time.time_ns()
                manifest["data_start_ns"] = manifest[
                    "producer_start_command_ns"
                ]
                internal_control.publish(
                    topics["producer_control"],
                    producer_command,
                )
                producer_started = wait_experiment_status(
                    internal_control,
                    "STARTED",
                    experiment_id,
                    local.get("status_timeout_s", 60),
                )
                manifest["producer_started"] = producer_started
                # Keep manifest updates in RAM during the active data phase.
                producer_done = wait_experiment_status(
                    internal_control,
                    "DONE",
                    experiment_id,
                    local.get("producer_timeout_s", 3600),
                )
                manifest["producer_done"] = producer_done
                manifest["producer_done_received_ns"] = time.time_ns()

                # Do not stop the gateway merely because the consumer has
                # been idle. First require proof that the gateway processed
                # all expected messages and persisted the RAM-buffered result.
                gateway_completion = local_run.wait_for_gateway_result(
                    float(experiment.get("drain_timeout_s", 180.0))
                )
                manifest["gateway_completion"] = gateway_completion

                drain_status = wait_for_consumer_drain(
                    cloud_control,
                    topics["consumer_control"],
                    experiment_id,
                    int(experiment["total_messages"]),
                    float(experiment.get("drain_idle_s", 5.0)),
                    float(experiment.get("drain_timeout_s", 180.0)),
                    float(experiment.get("drain_poll_s", 1.0)),
                )
                manifest["consumer_drain"] = drain_status
                cloud_control.publish(
                    topics["consumer_control"],
                    {
                        "command": "STOP",
                        "experiment_id": experiment_id,
                    },
                )
                consumer_stopped = wait_experiment_status(
                    cloud_control,
                    "STOPPED",
                    experiment_id,
                    local.get("status_timeout_s", 60),
                )
                manifest["consumer_stopped"] = consumer_stopped

                expected_messages = int(experiment["total_messages"])
                received_messages = int(
                    consumer_stopped.get("received_messages", 0)
                )
                if received_messages != expected_messages:
                    raise RuntimeError(
                        "Consumer message count mismatch: "
                        f"expected {expected_messages}, got {received_messages}"
                    )

                stop_status = local_run.stop()
                local_run = None
                manifest["local_stop_status"] = stop_status

                result_filename = (
                    "measurement_raw.csv"
                    if experiment["mode"] == "measurement"
                    else "gateway_timestamps.csv"
                )
                result_path = output_dir / result_filename
                if not result_path.exists():
                    raise RuntimeError(
                        f"Missing gateway result file after shutdown: {result_path}"
                    )

                with result_path.open("r", encoding="utf-8") as result_file:
                    result_line_count = sum(1 for _ in result_file)
                expected_line_count = int(experiment["total_messages"]) + 1
                if result_line_count != expected_line_count:
                    raise RuntimeError(
                        f"Unexpected row count in {result_path}: "
                        f"expected {expected_line_count}, got {result_line_count}"
                    )
                manifest["gateway_result_file"] = str(result_path)
                manifest["gateway_result_line_count"] = result_line_count

                if experiment.get("resource_monitor_enabled", False):
                    resource_path = output_dir / "resource_usage.csv"
                    if not resource_path.exists():
                        raise RuntimeError(
                            f"Missing resource result file after shutdown: {resource_path}"
                        )
                    manifest["resource_result_file"] = str(resource_path)

                manifest["status"] = "completed"
                manifest["automation_end_ns"] = time.time_ns()
                save_json_atomic(manifest_path, manifest)
                print(f"COMPLETED {experiment_id}", flush=True)

                time.sleep(experiment.get("cooldown_s", 5.0))

            except Exception as error:
                if local_run is not None:
                    local_run.stop()

                try:
                    cloud_control.publish(
                        topics["consumer_control"],
                        {
                            "command": "STOP",
                            "experiment_id": experiment_id,
                        },
                    )
                except Exception:
                    pass

                manifest["status"] = "failed"
                manifest["error"] = repr(error)
                manifest["automation_end_ns"] = time.time_ns()
                save_json_atomic(manifest_path, manifest)
                print(f"FAILED {experiment_id}: {error}", flush=True)

                if not args.continue_on_error:
                    raise

    finally:
        internal_control.stop()
        cloud_control.stop()


if __name__ == "__main__":
    main()
