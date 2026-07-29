# Experiment Sequence Flow

## Manual services

Before starting the automated experiment set:

```text
Producer host:
    start producer_runner.py
    → connect to the Internal Broker
    → wait for control commands

Consumer host:
    start cloud_client_runner.py
    → connect to the Cloud Broker
    → wait for control commands
```

These runners must be online because `automation.py` checks them using PING/PONG.

## Per-experiment order

The important run order is:

```text
automation.py selects one experiment
    ↓
run_experiment.py starts main.py on the Gateway
    ↓
Gateway initializes the Device Twin, models, crypto backend, and MQTT
    ↓
Gateway creates gateway.ready
    ↓
automation.py starts the Consumer and waits for READY
    ↓
automation.py sends START to the Producer
    ↓
Producer begins publishing data
```

Therefore, the Gateway application is ready before Producer data publishing starts.

## Per-message flow

```text
Producer records T1
    ↓
Producer publishes encrypted input
    ↓
Gateway records T2
    ↓
Decrypt → Sensor Analytics → AI@Edge → Encrypt
    ↓
Gateway records T3
    ↓
Gateway publishes processed output
    ↓
Consumer records T4
```

In Measurement Mode, stage boundaries are recorded with `time.perf_counter_ns()`.

In Deployment Mode, there are no stage-level timers. Only the distributed boundary timestamps are collected, with T2 and T3 recorded on the Gateway using `time.time_ns()`.

## Completion and persistence

```text
processed_messages reaches total_messages
    ↓
an in-memory completion Event is set
    ↓
main.py detects completion
    ↓
device_manager.flush_results()
    ↓
RAM-buffered rows are written to the final CSV
```

No per-message CSV write occurs inside the MQTT callback.
