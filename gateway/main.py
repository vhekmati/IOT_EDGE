#!/usr/bin/env python3

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

STOP_REQUESTED = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--local-config", required=True)
    parser.add_argument("--ready-file", required=True)
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


def signal_handler(signum, frame):
    del signum, frame
    global STOP_REQUESTED
    STOP_REQUESTED = True


def main():
    args = parse_args()
    experiment = load_json(args.experiment_config)
    local = load_json(args.local_config)

    runtime = local["runtime"]
    libs_dir = runtime["libs_dir"]

    # Keep this gateway directory ahead of the native-library directories so
    # the patched local DeviceManager/DeviceTwin modules are always imported.
    base_dir = str(Path(__file__).resolve().parent)
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
    for extra_path in (libs_dir, os.path.join(libs_dir, "pcd")):
        if extra_path not in sys.path:
            sys.path.append(extra_path)

    os.environ["LD_LIBRARY_PATH"] = (
        libs_dir + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    )

    from DeviceManager import DeviceManager
    from IoTGatewayPublicClient import IoTGatewayPublicClient

    output_dir = Path(experiment["output_directory"])
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment["runtime"] = runtime

    cloud = local["cloud_broker"]
    internal = local["internal_broker"]
    topics = local["topics"]

    public_client = IoTGatewayPublicClient({
        "client_id": (
            "gateway-public-" + experiment["experiment_id"]
        )[:60],
        "broker_ip": cloud["host"],
        "broker_port": cloud["port"],
        "ca_certs": cloud["ca_certs"],
        "certfile": cloud["certfile"],
        "keyfile": cloud["keyfile"],
    })
    public_client.start(
        timeout_s=local.get("gateway_start_timeout_s", 60)
    )

    device_manager = DeviceManager(public_client)
    device_manager.provision_device_twin(
        {"deviceTwinId": local.get("device_twin_id", "dtwin1")},
        {
            "broker_receiver_ip": internal["host"],
            "broker_receiver_port": internal["port"],
            "receiver_sub_topic": topics["data"],
            "receiver_ca_certs": internal["ca_certs"],
            "receiver_certfile": internal["certfile"],
            "receiver_keyfile": internal["keyfile"],
            "cloud_pub_topic": topics["data"],
        },
        {
            "processors": [
                "Decrypt",
                "DataAnalysis",
                "AIProcessing",
                "Encrypt",
            ],
            "experiment": experiment,
        },
    )

    device_manager.wait_until_ready(
        local.get("gateway_start_timeout_s", 60)
    )

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    ready_path = Path(args.ready_file)
    ready_path.write_text(
        experiment["experiment_id"],
        encoding="utf-8",
    )

    results_flushed = False

    # The main thread observes an in-memory completion event. Once all expected
    # messages have finished, it writes the buffered CSV outside the MQTT
    # callback and outside all per-message timing intervals. It then remains
    # alive so queued MQTT output can drain to the consumer.
    while not STOP_REQUESTED:
        if (
            not results_flushed
            and device_manager.all_processing_complete()
        ):
            device_manager.flush_results()
            results_flushed = True
        time.sleep(0.05)

    # Failure/early-stop fallback. The automation validates the row count, so
    # a partial file can never be accepted as a completed benchmark.
    if not results_flushed:
        device_manager.flush_results()

    device_manager.shutdown()
    device_manager.wait()

    # Do not call blocking loop_stop() methods during process teardown. The
    # process is dedicated to one experiment and the OS releases all resources.
    if public_client.client is not None:
        try:
            public_client.client.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
