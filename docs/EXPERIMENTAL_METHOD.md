# Experimental Method

## Experimental Design

The evaluation contains 12 backend combinations:

```text
2 Sensor Analytics × 2 AI × 3 Cryptography = 12 combinations
```

Each combination is executed three times. Every run contains 50 warm-up and 1,000 steady-state messages. Only `seq_id 51 ... 1050` is analyzed.

## Measurement Mode

Measurement Mode evaluates local Gateway stage and total latency at a safe publishing interval.

```python
time.perf_counter_ns()
```

is used in:

```text
EdgeProcessing.py → EdgeProcessingChain._process_measurement()
```

Only Gateway-side timestamp files are generated. Timing rows remain in RAM and are written after completion.

## Deployment Mode

Deployment Mode evaluates the distributed system at a near-saturation publishing interval. There is no stage-level timing.

```python
time.time_ns()
```

records:

```text
T1: Producer publish
T2: Gateway ingress
T3: Gateway egress-ready
T4: Consumer arrival
```

## Clock Synchronization

The Windows Producer, embedded Gateway, and Linux Cloud Consumer were synchronized against:

```text
time1.google.com
```

Same-host measurements are reliable. Cross-host latency is approximate because residual NTP offsets may remain.

## No File I/O in the Per-Message Path

During message processing, Measurement rows are appended to `self.measurement_rows` in RAM and Deployment rows to `self.gateway_rows`.

After all messages finish:

```text
main.py
  └── DeviceManager.flush_results()
       └── DeviceTwin.flush_results()
```

writes the buffered results. No per-message CSV write occurs inside the MQTT callback.
