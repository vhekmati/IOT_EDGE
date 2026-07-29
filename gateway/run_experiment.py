#!/usr/bin/env python3

import json
import os
import signal
import subprocess
import time
from pathlib import Path


class LocalGatewayRun:
    def __init__(self, experiment_config_path, local_config_path):
        self.base_dir = Path(__file__).resolve().parent
        self.experiment_config_path = Path(experiment_config_path).resolve()
        self.local_config_path = Path(local_config_path).resolve()

        with self.experiment_config_path.open(
            "r",
            encoding="utf-8",
        ) as input_file:
            self.experiment = json.load(input_file)

        with self.local_config_path.open(
            "r",
            encoding="utf-8",
        ) as input_file:
            self.local = json.load(input_file)

        self.output_dir = Path(self.experiment["output_directory"])
        if not self.output_dir.is_absolute():
            self.output_dir = self.base_dir / self.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.experiment["output_directory"] = str(self.output_dir)
        self.runtime_config_path = (
            self.output_dir / "runtime_experiment_config.json"
        )
        with self.runtime_config_path.open(
            "w",
            encoding="utf-8",
        ) as output_file:
            json.dump(self.experiment, output_file, indent=2)

        self.ready_file = self.output_dir / "gateway.ready"
        self.gateway_process = None
        self.monitor_process = None
        self.gateway_output_stream = None

    def start(self):
        if self.ready_file.exists():
            self.ready_file.unlink()

        python_executable = self.local.get(
            "python_executable",
            "python3",
        )
        # Native/Python stdout and stderr handling is configurable.
        # Keep gateway_log_enabled=false for final benchmark runs: output is
        # discarded and no log file is written during the timed run.
        # Enable it only for a separate diagnostic run; all process output
        # (not only TitanSSL) is then written under the experiment directory.
        if self.experiment.get("gateway_log_enabled", False):
            log_name = self.experiment.get(
                "gateway_log_filename",
                "gateway_native.log",
            )
            log_path = Path(log_name)
            if not log_path.is_absolute():
                log_path = self.output_dir / log_path
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self.gateway_output_stream = log_path.open(
                "w",
                encoding="utf-8",
            )
            process_stdout = self.gateway_output_stream
            process_stderr = subprocess.STDOUT
        else:
            process_stdout = subprocess.DEVNULL
            process_stderr = subprocess.DEVNULL

        self.gateway_process = subprocess.Popen(
            [
                python_executable,
                str(self.base_dir / "main.py"),
                "--experiment-config",
                str(self.runtime_config_path),
                "--local-config",
                str(self.local_config_path),
                "--ready-file",
                str(self.ready_file),
            ],
            cwd=str(self.base_dir),
            stdout=process_stdout,
            stderr=process_stderr,
            start_new_session=True,
        )

        timeout_s = self.local.get("gateway_start_timeout_s", 60)
        deadline = time.time() + timeout_s

        while time.time() < deadline:
            if self.ready_file.exists():
                break
            if self.gateway_process.poll() is not None:
                raise RuntimeError(
                    "Gateway exited before becoming ready"
                )
            time.sleep(0.2)
        else:
            self.stop()
            raise TimeoutError("Gateway readiness timed out")

        # Strict timing runs disable resource monitoring by default because
        # reading /proc is additional activity on the gateway. Enable it only
        # for a separate resource-profiling run.
        if self.experiment.get("resource_monitor_enabled", False):
            self.monitor_process = subprocess.Popen(
                [
                    python_executable,
                    str(self.base_dir / "monitor_res.py"),
                    "--pid",
                    str(self.gateway_process.pid),
                    "--interval",
                    str(
                        self.experiment.get(
                            "resource_sampling_interval_s",
                            0.5,
                        )
                    ),
                    "--output",
                    str(self.output_dir / "resource_usage.csv"),
                    "--experiment-id",
                    self.experiment["experiment_id"],
                ],
                cwd=str(self.base_dir),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

    @staticmethod
    def _stop_process(process, timeout_s):
        status = {
            "started": process is not None,
            "forced_kill": False,
            "returncode": None,
        }
        if process is None:
            return status
        if process.poll() is not None:
            status["returncode"] = process.returncode
            return status

        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        try:
            process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            status["forced_kill"] = True
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            process.wait()

        status["returncode"] = process.returncode
        return status


    def wait_for_gateway_result(self, timeout_s):
        result_filename = (
            "measurement_raw.csv"
            if self.experiment["mode"] == "measurement"
            else "gateway_timestamps.csv"
        )
        result_path = self.output_dir / result_filename
        expected_line_count = int(self.experiment["total_messages"]) + 1
        deadline = time.monotonic() + timeout_s

        while time.monotonic() < deadline:
            if self.gateway_process.poll() is not None:
                raise RuntimeError(
                    "Gateway exited before persisting the result file"
                )

            if result_path.exists():
                with result_path.open("r", encoding="utf-8") as result_file:
                    line_count = sum(1 for _ in result_file)
                if line_count == expected_line_count:
                    return {
                        "result_path": str(result_path),
                        "line_count": line_count,
                        "completed_ns": time.time_ns(),
                    }
                raise RuntimeError(
                    f"Gateway result has {line_count} lines; "
                    f"expected {expected_line_count}: {result_path}"
                )

            time.sleep(0.1)

        raise TimeoutError(
            "Gateway did not process and persist all expected messages "
            f"within {timeout_s} seconds"
        )

    def stop(self):
        # Stop monitoring before gateway shutdown so resource statistics do not
        # include CSV flushing, MQTT disconnect, or backend teardown.
        monitor_status = self._stop_process(self.monitor_process, 10)
        gateway_status = self._stop_process(
            self.gateway_process,
            self.local.get("gateway_shutdown_timeout_s", 10),
        )

        if self.gateway_output_stream is not None:
            self.gateway_output_stream.close()
            self.gateway_output_stream = None

        return {
            "monitor": monitor_status,
            "gateway": gateway_status,
        }

    @property
    def gateway_pid(self):
        if self.gateway_process is None:
            return None
        return self.gateway_process.pid
