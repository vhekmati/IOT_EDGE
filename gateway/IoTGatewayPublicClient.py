import ssl
import threading

import paho.mqtt.client as mqtt


class IoTGatewayPublicClient:
    def __init__(self, config):
        self.config = config
        self.client = None
        self.connected = threading.Event()
        self.publish_lock = threading.Lock()

    def _on_connect(self, client, userdata, flags, rc):
        del client, userdata, flags
        if rc == 0:
            self.connected.set()

    def start(self, timeout_s=30):
        self.client = mqtt.Client(
            client_id=self.config.get(
                "client_id",
                "iot-gateway-public-client",
            ),
            clean_session=True,
            protocol=mqtt.MQTTv311,
        )
        self.client.on_connect = self._on_connect
        self.client.tls_set(
            ca_certs=self.config["ca_certs"],
            certfile=self.config["certfile"],
            keyfile=self.config["keyfile"],
            cert_reqs=ssl.CERT_REQUIRED,
        )
        self.client.connect(
            self.config["broker_ip"],
            self.config["broker_port"],
        )
        self.client.loop_start()

        if not self.connected.wait(timeout_s):
            raise TimeoutError("Cloud MQTT connection timed out")

    def publish(self, topic, payload, qos=0):
        with self.publish_lock:
            return self.client.publish(topic, payload, qos=qos)

    def stop(self):
        if self.client is None:
            return
        self.client.loop_stop()
        self.client.disconnect()
        self.connected.clear()
