# Automation Workflow

## Complete Sequence

```text
1. Producer Runner and Consumer Runner wait for control commands.
2. automation.py selects the next experiment JSON.
3. run_experiment.py creates the runtime config and starts main.py.
4. main.py initializes MQTT, Device Manager, and Device Twin.
5. The Device Twin initializes Sensor Analytics, AI, and cryptographic backends.
6. main.py creates gateway.ready.
7. automation.py starts the Consumer and waits for READY.
8. automation.py sends START to the Producer.
9. The Producer begins publishing.
10. The Gateway processes and republishes messages.
11. Timing rows remain in RAM.
12. Results are written after completion.
13. automation.py validates the run and starts the next experiment.
```

## Main Automation Files

| File | Responsibility |
|---|---|
| `automation.py` | Selects configs, performs PING/PONG checks, coordinates START/STOP, and validates completion |
| `run_experiment.py` | Creates runtime config, starts `main.py`, waits for readiness, starts monitoring, and checks outputs |
| `main.py` | Initializes the Gateway, creates `gateway.ready`, detects completion, and flushes results |
| `DeviceTwin.py` | Receives messages, executes the chain, buffers Deployment timestamps, and publishes output |
| `EdgeProcessing.py` | Implements the chain and Measurement timing |

## Runtime Files

| File | Created | Purpose |
|---|---|---|
| `runtime_experiment_config.json` | Before `main.py` | Runtime experiment configuration |
| `run_manifest.json` | Before and after run | Run status and errors |
| `gateway.ready` | After initialization | Gateway readiness marker |
| Result CSV | After completion | Buffered timing results |
| `resource_usage.csv` | After monitor stops | CPU and RAM samples |

## Resource Monitoring

When `resource_monitor_enabled` is `true`, `run_experiment.py` starts `main.py`, waits for `gateway.ready`, and launches:

```text
monitor_res.py --pid <gateway_pid>
```

The monitor samples the Gateway process and its child processes. Samples are buffered and written after monitoring stops.
