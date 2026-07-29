# Implementation Notes

## Gateway Structure

```text
main.py
  ↓
DeviceManager.py
  ↓
DeviceTwin.py
  ↓
EdgeProcessing.py
```

| Component | Responsibility |
|---|---|
| `main.py` | Creates the Gateway application and readiness marker |
| `DeviceManager.py` | Creates and manages Device Twins |
| `DeviceTwin.py` | Receives messages, executes the chain, and publishes output |
| `EdgeProcessing.py` | Implements Decrypt, Sensor Analytics, AI@Edge, Encrypt, and Measurement timing |
| `IoTGatewayPublicClient.py` | Publishes results to the cloud broker |

## OP-TEE Backend

The final OP-TEE backend uses direct OpenSSL EVP API calls through `libcrypto` and the TitanSSL Engine.

For each Device Twin:

- the Engine is initialized once;
- one `CyberSecurityProcessor` is created;
- `DecryptProcessor` and `EncryptProcessor` share it;
- EVP contexts perform encryption and decryption.

```json
"crypto_backend": "optee"
```

## Main Experiment Fields

| Field | Meaning |
|---|---|
| `experiment_id` | Unique run identifier |
| `combination_id` | Backend combination |
| `run_number` | Independent repetition |
| `mode` | `measurement` or `deployment` |
| `analytics_backend` | `python` or `numpy` |
| `ai_backend` | `cpu` or `edgetpu` |
| `crypto_backend` | `pycryptodome`, `openssl`, or `optee` |
| `publish_interval_s` | Producer publishing interval |
| `warmup_messages` | Warm-up count |
| `measured_messages` | Steady-state count |
| `total_messages` | Total message count |
| `output_directory` | Gateway result directory |

Commit only example local configs. Do not commit private keys, real certificates, AES keys, or personal absolute paths.
