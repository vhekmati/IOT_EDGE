import threading
import time

from DeviceTwin import DeviceTwin


class DeviceManager:
    def __init__(self, public_client):
        self.public_client = public_client
        self.device_twins = []
        self.threads = []
        self.errors = []
        self.lock = threading.Lock()

    def provision_device_twin(
        self,
        device_config,
        connection_config,
        chain_config,
    ):
        thread = threading.Thread(
            target=self._run_twin,
            args=(device_config, connection_config, chain_config),
            name=f"Thread-{device_config['deviceTwinId']}",
        )
        thread.start()
        self.threads.append(thread)

    def _run_twin(self, device_config, connection_config, chain_config):
        try:
            twin = DeviceTwin(device_config, self.public_client)
            twin.start_edge_processing_chain(chain_config)
            twin.start_connections(connection_config)
            with self.lock:
                self.device_twins.append(twin)
        except Exception as error:
            with self.lock:
                self.errors.append(error)

    def wait_until_ready(self, timeout_s):
        deadline = time.time() + timeout_s

        while time.time() < deadline:
            with self.lock:
                if self.errors:
                    raise self.errors[0]
                twins = list(self.device_twins)

            if twins and all(twin.connected.is_set() for twin in twins):
                return

            time.sleep(0.1)

        raise TimeoutError("Device Twin did not become ready")

    def all_processing_complete(self):
        with self.lock:
            if self.errors:
                raise self.errors[0]
            twins = list(self.device_twins)

        return bool(twins) and all(
            twin.is_processing_complete() for twin in twins
        )

    def flush_results(self):
        with self.lock:
            twins = list(self.device_twins)

        for twin in twins:
            twin.flush_results()

    def shutdown(self):
        with self.lock:
            twins = list(self.device_twins)

        for twin in twins:
            twin.stop()

    def wait(self):
        for thread in self.threads:
            thread.join(timeout=5)
