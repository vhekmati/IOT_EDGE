# Data Analysis

## Measurement Analysis

Expected input:

```text
gateway_results/<experiment_id>/
├── runtime_experiment_config.json
├── measurement_raw.csv
└── resource_usage.csv
```

No Producer-side or Consumer-side timestamp files are generated.

```bash
python analysis/analyze_measurement.py --gateway-results gateway_results --output analysis_output/measurement
```

The script removes warm-up rows, calculates stage, crypto, total, and unattributed latency, and aggregates three runs using mean and sample SD.

```text
stage_ms = (stage_end_ns - stage_start_ns) / 1,000,000
total_ms = (total_end_ns - total_start_ns) / 1,000,000
crypto_ms = decrypt_ms + encrypt_ms
unattributed_ms = total_ms - decrypt_ms - analytics_ms - ai_ms - encrypt_ms
```

## Deployment Analysis

Expected input:

```text
gateway_results/<experiment_id>/gateway_timestamps.csv
producer_results/<experiment_id>/producer_timestamps.csv
consumer_results/<experiment_id>/consumer_timestamps.csv
```

```bash
python analysis/analyze_deployment.py --gateway-results gateway_results --producer-results producer_results --consumer-results consumer_results --output analysis_output/deployment
```

The three files are matched using:

```text
experiment_id + seq_id
```

The merged table is used for:

```text
T2 - T1
T4 - T3
T4 - T1
```

Gateway latency is calculated locally:

```text
T3 - T2
```

Per-host throughput is:

```text
N × 1,000,000,000 / (last_timestamp_ns - first_timestamp_ns)
```

The scripts create run summaries, case summaries, professor tables, rankings, detailed tables, findings, and figures.
