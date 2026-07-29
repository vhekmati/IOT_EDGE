#!/usr/bin/env python3

"""
Analyze all final Deployment-mode runs.

Expected input structure:

gateway_results/
    final_case_01_deployment_run_01/
        runtime_experiment_config.json
        gateway_timestamps.csv
        resource_usage.csv

producer_results/
    final_case_01_deployment_run_01/
        producer_timestamps.csv

consumer_results/
    final_case_01_deployment_run_01/
        consumer_timestamps.csv

Main outputs:
    deployment_run_summary.csv       one row per run (36 rows)
    deployment_case_summary.csv      one row per combination (12 rows)
    deployment_professor_table.csv   concise presentation table
    deployment_ranking.csv           numeric table sorted by throughput
    deployment_detailed_table.csv    detailed numeric statistics
    deployment_key_findings.txt
    figures/*.png

Only steady-state messages are analyzed.
With 50 warm-up messages, this means seq_id 51 ... 1050.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NS_TO_MS = 1_000_000.0
NS_TO_SECONDS = 1_000_000_000.0
CSV_ENCODING = "utf-8-sig"

BACKEND_NAMES = {
    "python": "Pure Python",
    "numpy": "NumPy",
    "cpu": "TFLite CPU",
    "edgetpu": "EdgeTPU",
    "pycryptodome": "PyCryptodome",
    "openssl": "OpenSSL",
    "optee": "OP-TEE",
}


def percentile_95(values):
    return float(np.percentile(values, 95))


def percentile_99(values):
    return float(np.percentile(values, 99))


def readable_backend_name(name):
    return BACKEND_NAMES.get(name, name)


def short_case_name(case_id):
    return case_id.replace("case_", "C").upper()


def read_config(run_dir):
    with (run_dir / "runtime_experiment_config.json").open(
        "r", encoding="utf-8"
    ) as file:
        return json.load(file)


def find_timestamp_column(data, prefix):
    """
    Return the timestamp column starting with t1_, t2_, t3_, or t4_.
    """
    return next(
        column for column in data.columns
        if column.startswith(prefix)
    )


def throughput_messages_per_second(data, timestamp_column):
    """
    Throughput formula used in this analysis:

        throughput = N / elapsed_time

    N is the number of steady-state messages observed on one host.
    elapsed_time is the difference between the last and first timestamps
    recorded on that same host.

    Since timestamps are in nanoseconds:

        throughput =
            N * 1,000,000,000
            -----------------
              last_ns - first_ns

    In the final experiment, N is normally 1000.
    """
    elapsed_ns = (
        data[timestamp_column].max()
        - data[timestamp_column].min()
    )

    return len(data) * NS_TO_SECONDS / elapsed_ns


def summarize_resource_usage(run_dir):
    resource = pd.read_csv(run_dir / "resource_usage.csv")

    return {
        "cpu_mean_percent": resource["cpu_percent_total"].mean(),
        "cpu_p95_percent": percentile_95(
            resource["cpu_percent_total"]
        ),
        "cpu_max_percent": resource["cpu_percent_total"].max(),
        "ram_mean_mb": resource["rss_mb_total"].mean(),
        "ram_max_mb": resource["rss_mb_total"].max(),
    }


def summarize_run(gateway_run, producer_run, consumer_run):
    """
    Calculate all Deployment KPIs for one run.
    """
    config = read_config(gateway_run)
    warmup_messages = int(config["warmup_messages"])

    gateway = pd.read_csv(
        gateway_run / "gateway_timestamps.csv"
    )
    producer = pd.read_csv(
        producer_run / "producer_timestamps.csv"
    )
    consumer = pd.read_csv(
        consumer_run / "consumer_timestamps.csv"
    )

    for data in [gateway, producer, consumer]:
        data["seq_id"] = data["seq_id"].astype(int)

    # Remove warm-up messages.
    gateway = gateway[
        gateway["seq_id"] > warmup_messages
    ].copy()
    producer = producer[
        producer["seq_id"] > warmup_messages
    ].copy()
    consumer = consumer[
        consumer["seq_id"] > warmup_messages
    ].copy()

    t1 = find_timestamp_column(producer, "t1_")
    t2 = find_timestamp_column(gateway, "t2_")
    t3 = find_timestamp_column(gateway, "t3_")
    t4 = find_timestamp_column(consumer, "t4_")

    # Gateway boundary latency:
    #
    # gateway_latency_ms = (T3 - T2) / 1,000,000
    #
    # T2 and T3 are both recorded on the Gateway, therefore this
    # latency is not affected by clock offsets between different hosts.
    gateway["gateway_latency_ms"] = (
        gateway[t3] - gateway[t2]
    ) / NS_TO_MS

    # Match the same message across Producer, Gateway, and Consumer.
    merged = producer[
        ["experiment_id", "seq_id", t1]
    ].merge(
        gateway[
            ["experiment_id", "seq_id", t2, t3]
        ],
        on=["experiment_id", "seq_id"],
    )

    merged = merged.merge(
        consumer[
            ["experiment_id", "seq_id", t4]
        ],
        on=["experiment_id", "seq_id"],
    )

    # Approximate Producer-to-Gateway latency:
    #
    # producer_to_gateway_ms = (T2 - T1) / 1,000,000
    #
    # T1 and T2 are recorded on different hosts, so this metric is
    # affected by the Producer-Gateway NTP clock-offset difference.
    merged["producer_to_gateway_ms"] = (
        merged[t2] - merged[t1]
    ) / NS_TO_MS

    # Reliable Gateway boundary latency:
    #
    # gateway_latency_ms = (T3 - T2) / 1,000,000
    #
    # This was already calculated from two timestamps on the Gateway.
    # It is not affected by inter-host clock offsets.

    # Approximate Gateway-to-Consumer latency:
    #
    # gateway_to_consumer_ms = (T4 - T3) / 1,000,000
    #
    # T3 and T4 are recorded on different hosts, so this metric is
    # affected by the Gateway-Consumer NTP clock-offset difference.
    merged["gateway_to_consumer_ms"] = (
        merged[t4] - merged[t3]
    ) / NS_TO_MS

    # Approximate end-to-end latency:
    #
    # approximate_e2e_ms = (T4 - T1) / 1,000,000
    #
    # T1 and T4 are recorded on different hosts. The result is therefore
    # affected by NTP clock-offset uncertainty. It is not corrected using
    # offsets measured after the experiments.
    merged["approx_e2e_ms"] = (
        merged[t4] - merged[t1]
    ) / NS_TO_MS

    producer_sequences = set(producer["seq_id"])
    consumer_sequences = set(consumer["seq_id"])

    missing_at_consumer = len(
        producer_sequences - consumer_sequences
    )

    # Message loss:
    #
    # loss_percent =
    #     (sent_messages - received_messages)
    #     ------------------------------------ * 100
    #                sent_messages
    loss_percent = (
        missing_at_consumer
        / len(producer_sequences)
        * 100.0
    )

    summary = {
        "experiment_id": config["experiment_id"],
        "case_id": config["combination_id"],
        "run_number": int(config["run_number"]),
        "analytics_backend": config["analytics_backend"],
        "ai_backend": config["ai_backend"],
        "crypto_backend": config["crypto_backend"],

        "producer_messages": len(producer),
        "gateway_messages": len(gateway),
        "consumer_messages": len(consumer),
        "matched_messages": len(merged),
        "missing_at_consumer": missing_at_consumer,
        "loss_percent": loss_percent,

        "producer_offered_rate_msg_s":
            throughput_messages_per_second(
                producer, t1
            ),

        "gateway_ingress_rate_msg_s":
            throughput_messages_per_second(
                gateway, t2
            ),

        "gateway_egress_rate_msg_s":
            throughput_messages_per_second(
                gateway, t3
            ),

        "consumer_throughput_msg_s":
            throughput_messages_per_second(
                consumer, t4
            ),

        "producer_to_gateway_ms_mean":
            merged["producer_to_gateway_ms"].mean(),

        "producer_to_gateway_ms_median":
            merged["producer_to_gateway_ms"].median(),

        "producer_to_gateway_ms_p95":
            percentile_95(
                merged["producer_to_gateway_ms"]
            ),

        "producer_to_gateway_ms_p99":
            percentile_99(
                merged["producer_to_gateway_ms"]
            ),

        "negative_producer_to_gateway_values": int(
            (merged["producer_to_gateway_ms"] < 0).sum()
        ),

        "gateway_latency_ms_mean":
            gateway["gateway_latency_ms"].mean(),

        "gateway_latency_ms_median":
            gateway["gateway_latency_ms"].median(),

        "gateway_latency_ms_p95":
            percentile_95(
                gateway["gateway_latency_ms"]
            ),

        "gateway_latency_ms_p99":
            percentile_99(
                gateway["gateway_latency_ms"]
            ),

        "gateway_to_consumer_ms_mean":
            merged["gateway_to_consumer_ms"].mean(),

        "gateway_to_consumer_ms_median":
            merged["gateway_to_consumer_ms"].median(),

        "gateway_to_consumer_ms_p95":
            percentile_95(
                merged["gateway_to_consumer_ms"]
            ),

        "gateway_to_consumer_ms_p99":
            percentile_99(
                merged["gateway_to_consumer_ms"]
            ),

        "negative_gateway_to_consumer_values": int(
            (merged["gateway_to_consumer_ms"] < 0).sum()
        ),

        "approx_e2e_ms_mean":
            merged["approx_e2e_ms"].mean(),

        "approx_e2e_ms_median":
            merged["approx_e2e_ms"].median(),

        "approx_e2e_ms_p95":
            percentile_95(
                merged["approx_e2e_ms"]
            ),

        "approx_e2e_ms_p99":
            percentile_99(
                merged["approx_e2e_ms"]
            ),

        "negative_approx_e2e_values": int(
            (merged["approx_e2e_ms"] < 0).sum()
        ),
    }

    # Throughput efficiency:
    #
    # efficiency_percent =
    #     consumer_throughput
    #     ------------------- * 100
    #     producer_offered_rate
    summary["throughput_efficiency_percent"] = (
        summary["consumer_throughput_msg_s"]
        / summary["producer_offered_rate_msg_s"]
        * 100.0
    )

    summary.update(
        summarize_resource_usage(gateway_run)
    )

    return summary


def create_case_summary(run_summary):
    """
    Combine the three independent runs of each backend combination.

    For every KPI:
        case_mean = mean(run_1, run_2, run_3)
        case_SD   = sample SD(run_1, run_2, run_3)
    """
    group_columns = [
        "case_id",
        "analytics_backend",
        "ai_backend",
        "crypto_backend",
    ]

    numeric_columns = [
        column
        for column in run_summary.columns
        if column not in group_columns + ["experiment_id"]
    ]

    case_summary = run_summary.groupby(
        group_columns
    )[numeric_columns].agg(["mean", "std"])

    case_summary.columns = [
        f"{metric}_{statistic}"
        for metric, statistic in case_summary.columns
    ]

    return case_summary.reset_index()


def create_numeric_ranking(case_summary):
    """
    Create the main numeric table sorted by highest Consumer throughput.

    Mean and SD remain in separate numeric columns so Excel can filter,
    sort, calculate, and plot them.
    """
    table = case_summary[
        [
            "case_id",
            "analytics_backend",
            "ai_backend",
            "crypto_backend",

            "consumer_throughput_msg_s_mean",
            "consumer_throughput_msg_s_std",

            "producer_offered_rate_msg_s_mean",
            "gateway_ingress_rate_msg_s_mean",
            "gateway_egress_rate_msg_s_mean",
            "throughput_efficiency_percent_mean",

            "producer_to_gateway_ms_mean_mean",
            "producer_to_gateway_ms_mean_std",
            "producer_to_gateway_ms_p95_mean",
            "producer_to_gateway_ms_p95_std",

            "gateway_latency_ms_mean_mean",
            "gateway_latency_ms_mean_std",
            "gateway_latency_ms_p95_mean",
            "gateway_latency_ms_p95_std",

            "gateway_to_consumer_ms_mean_mean",
            "gateway_to_consumer_ms_mean_std",
            "gateway_to_consumer_ms_p95_mean",
            "gateway_to_consumer_ms_p95_std",

            "approx_e2e_ms_mean_mean",
            "approx_e2e_ms_mean_std",
            "approx_e2e_ms_p95_mean",
            "approx_e2e_ms_p95_std",

            "loss_percent_mean",
            "negative_approx_e2e_values_mean",

            "cpu_mean_percent_mean",
            "ram_mean_mb_mean",
        ]
    ].copy()

    table.columns = [
        "Case",
        "Analytics",
        "AI",
        "Crypto",

        "Consumer throughput mean (msg/s)",
        "Consumer throughput SD (msg/s)",

        "Producer offered rate (msg/s)",
        "Gateway ingress rate (msg/s)",
        "Gateway egress rate (msg/s)",
        "Delivered-to-offered rate ratio (%)",

        "Producer-to-Gateway mean (ms)",
        "Producer-to-Gateway mean SD (ms)",
        "Producer-to-Gateway p95 mean (ms)",
        "Producer-to-Gateway p95 SD (ms)",

        "Gateway latency mean (ms)",
        "Gateway latency SD (ms)",
        "Gateway p95 mean (ms)",
        "Gateway p95 SD (ms)",

        "Gateway-to-Consumer mean (ms)",
        "Gateway-to-Consumer mean SD (ms)",
        "Gateway-to-Consumer p95 mean (ms)",
        "Gateway-to-Consumer p95 SD (ms)",

        "Approx. E2E mean (ms)",
        "Approx. E2E mean SD (ms)",
        "Approx. E2E p95 mean (ms)",
        "Approx. E2E p95 SD (ms)",

        "Loss (%)",
        "Mean negative E2E count",

        "CPU mean (%)",
        "RAM mean (MB)",
    ]

    table["Analytics"] = table[
        "Analytics"
    ].map(readable_backend_name)

    table["AI"] = table[
        "AI"
    ].map(readable_backend_name)

    table["Crypto"] = table[
        "Crypto"
    ].map(readable_backend_name)

    table["Case"] = table[
        "Case"
    ].map(short_case_name)

    table = table.sort_values(
        "Consumer throughput mean (msg/s)",
        ascending=False,
    ).reset_index(drop=True)

    table.insert(
        0,
        "Throughput rank",
        range(1, len(table) + 1),
    )

    numeric_columns = table.select_dtypes(
        include="number"
    ).columns

    table[numeric_columns] = table[
        numeric_columns
    ].round(3)

    return table


def create_professor_table(numeric_ranking):
    """
    Create a concise presentation table.

    Consumer throughput, Gateway latency, and approximate E2E latency
    are shown as mean ± SD across the three independent runs.
    """
    table = numeric_ranking.copy()

    table["Consumer throughput mean ± SD (msg/s)"] = table.apply(
        lambda row: (
            f"{row['Consumer throughput mean (msg/s)']:.3f} ± "
            f"{row['Consumer throughput SD (msg/s)']:.3f}"
        ),
        axis=1,
    )

    # Cross-host latency values are displayed in seconds in the concise
    # professor table because they can reach several thousand milliseconds.
    table["Producer-to-Gateway mean ± SD (s)"] = table.apply(
        lambda row: (
            f"{row['Producer-to-Gateway mean (ms)'] / 1000:.3f} ± "
            f"{row['Producer-to-Gateway mean SD (ms)'] / 1000:.3f}"
        ),
        axis=1,
    )

    table["Producer-to-Gateway p95 ± SD (s)"] = table.apply(
        lambda row: (
            f"{row['Producer-to-Gateway p95 mean (ms)'] / 1000:.3f} ± "
            f"{row['Producer-to-Gateway p95 SD (ms)'] / 1000:.3f}"
        ),
        axis=1,
    )

    table["Gateway latency mean ± SD (ms)"] = table.apply(
        lambda row: (
            f"{row['Gateway latency mean (ms)']:.3f} ± "
            f"{row['Gateway latency SD (ms)']:.3f}"
        ),
        axis=1,
    )

    table["Gateway p95 mean ± SD (ms)"] = table.apply(
        lambda row: (
            f"{row['Gateway p95 mean (ms)']:.3f} ± "
            f"{row['Gateway p95 SD (ms)']:.3f}"
        ),
        axis=1,
    )

    table["Gateway-to-Consumer mean ± SD (s)"] = table.apply(
        lambda row: (
            f"{row['Gateway-to-Consumer mean (ms)'] / 1000:.3f} ± "
            f"{row['Gateway-to-Consumer mean SD (ms)'] / 1000:.3f}"
        ),
        axis=1,
    )

    table["Gateway-to-Consumer p95 ± SD (s)"] = table.apply(
        lambda row: (
            f"{row['Gateway-to-Consumer p95 mean (ms)'] / 1000:.3f} ± "
            f"{row['Gateway-to-Consumer p95 SD (ms)'] / 1000:.3f}"
        ),
        axis=1,
    )

    table["Approx. E2E mean ± SD (s)"] = table.apply(
        lambda row: (
            f"{row['Approx. E2E mean (ms)'] / 1000:.3f} ± "
            f"{row['Approx. E2E mean SD (ms)'] / 1000:.3f}"
        ),
        axis=1,
    )

    table["Approx. E2E p95 ± SD (s)"] = table.apply(
        lambda row: (
            f"{row['Approx. E2E p95 mean (ms)'] / 1000:.3f} ± "
            f"{row['Approx. E2E p95 SD (ms)'] / 1000:.3f}"
        ),
        axis=1,
    )

    return table[
        [
            "Throughput rank",
            "Case",
            "Analytics",
            "AI",
            "Crypto",

            "Consumer throughput mean ± SD (msg/s)",
            "Producer offered rate (msg/s)",
            "Gateway ingress rate (msg/s)",
            "Gateway egress rate (msg/s)",
            "Delivered-to-offered rate ratio (%)",

            "Producer-to-Gateway mean ± SD (s)",
            "Producer-to-Gateway p95 ± SD (s)",

            "Gateway latency mean ± SD (ms)",
            "Gateway p95 mean ± SD (ms)",

            "Gateway-to-Consumer mean ± SD (s)",
            "Gateway-to-Consumer p95 ± SD (s)",

            "Approx. E2E mean ± SD (s)",
            "Approx. E2E p95 ± SD (s)",

            "Loss (%)",
            "CPU mean (%)",
            "RAM mean (MB)",
        ]
    ]


def create_detailed_table(case_summary):
    """
    Create a detailed numeric table for report writing.
    """
    columns = [
        "case_id",
        "analytics_backend",
        "ai_backend",
        "crypto_backend",

        "consumer_throughput_msg_s_mean",
        "consumer_throughput_msg_s_std",

        "producer_offered_rate_msg_s_mean",
        "producer_offered_rate_msg_s_std",

        "gateway_ingress_rate_msg_s_mean",
        "gateway_ingress_rate_msg_s_std",

        "gateway_egress_rate_msg_s_mean",
        "gateway_egress_rate_msg_s_std",

        "throughput_efficiency_percent_mean",
        "throughput_efficiency_percent_std",

        "producer_to_gateway_ms_mean_mean",
        "producer_to_gateway_ms_mean_std",
        "producer_to_gateway_ms_median_mean",
        "producer_to_gateway_ms_median_std",
        "producer_to_gateway_ms_p95_mean",
        "producer_to_gateway_ms_p95_std",
        "producer_to_gateway_ms_p99_mean",
        "producer_to_gateway_ms_p99_std",

        "gateway_latency_ms_mean_mean",
        "gateway_latency_ms_mean_std",

        "gateway_latency_ms_median_mean",
        "gateway_latency_ms_median_std",

        "gateway_latency_ms_p95_mean",
        "gateway_latency_ms_p95_std",

        "gateway_latency_ms_p99_mean",
        "gateway_latency_ms_p99_std",

        "gateway_to_consumer_ms_mean_mean",
        "gateway_to_consumer_ms_mean_std",
        "gateway_to_consumer_ms_median_mean",
        "gateway_to_consumer_ms_median_std",
        "gateway_to_consumer_ms_p95_mean",
        "gateway_to_consumer_ms_p95_std",
        "gateway_to_consumer_ms_p99_mean",
        "gateway_to_consumer_ms_p99_std",

        "approx_e2e_ms_mean_mean",
        "approx_e2e_ms_mean_std",

        "approx_e2e_ms_median_mean",
        "approx_e2e_ms_median_std",

        "approx_e2e_ms_p95_mean",
        "approx_e2e_ms_p95_std",

        "approx_e2e_ms_p99_mean",
        "approx_e2e_ms_p99_std",

        "loss_percent_mean",
        "loss_percent_std",

        "negative_approx_e2e_values_mean",

        "cpu_mean_percent_mean",
        "cpu_mean_percent_std",
        "cpu_p95_percent_mean",

        "ram_mean_mb_mean",
        "ram_mean_mb_std",
        "ram_max_mb_mean",
    ]

    table = case_summary[columns].copy()

    table = table.sort_values(
        "consumer_throughput_msg_s_mean",
        ascending=False,
    ).reset_index(drop=True)

    table.insert(
        0,
        "throughput_rank",
        range(1, len(table) + 1),
    )

    numeric_columns = table.select_dtypes(
        include="number"
    ).columns

    table[numeric_columns] = table[
        numeric_columns
    ].round(3)

    return table


def add_case_labels(data):
    """
    Add a readable full backend label.

    Example:
        C11 | NumPy / EdgeTPU / OpenSSL
    """
    data = data.copy()

    data["case_label"] = data.apply(
        lambda row: (
            f"{short_case_name(row['case_id'])} | "
            f"{readable_backend_name(row['analytics_backend'])} / "
            f"{readable_backend_name(row['ai_backend'])} / "
            f"{readable_backend_name(row['crypto_backend'])}"
        ),
        axis=1,
    )

    return data


def deployment_order(case_summary):
    """
    Use one consistent order for all Deployment charts:
    highest Consumer throughput to lowest Consumer throughput.
    """
    data = case_summary.sort_values(
        "consumer_throughput_msg_s_mean",
        ascending=False,
    ).reset_index(drop=True)

    return add_case_labels(data)


def add_value_labels(
    ax,
    bars,
    means,
    standard_deviations,
    suffix,
):
    """
    Write mean ± SD at the end of each horizontal bar.
    """
    largest_value = max(means) if len(means) else 1
    label_gap = largest_value * 0.015

    for bar, mean, sd in zip(
        bars,
        means,
        standard_deviations,
    ):
        sd = 0.0 if pd.isna(sd) else sd

        ax.text(
            mean + sd + label_gap,
            bar.get_y() + bar.get_height() / 2,
            f"{mean:.2f} ± {sd:.2f} {suffix}",
            va="center",
            fontsize=8,
        )


def save_horizontal_bar_chart(
    case_summary,
    mean_column,
    sd_column,
    xlabel,
    title,
    path,
    suffix,
    note=None,
):
    """
    Save a readable horizontal bar chart.

    Error bars show SD across the three independent runs.
    """
    data = deployment_order(case_summary)

    means = data[mean_column].to_numpy()
    standard_deviations = data[
        sd_column
    ].fillna(0).to_numpy()

    figure_height = max(
        7,
        len(data) * 0.65,
    )

    fig, ax = plt.subplots(
        figsize=(15, figure_height)
    )

    bars = ax.barh(
        data["case_label"],
        means,
        xerr=standard_deviations,
        capsize=4,
    )

    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Backend combination")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)

    add_value_labels(
        ax,
        bars,
        means,
        standard_deviations,
        suffix,
    )

    largest = max(
        means + standard_deviations
    )

    ax.set_xlim(
        0,
        largest * 1.30,
    )

    if note:
        ax.text(
            0.98,
            0.98,
            note,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "alpha": 0.85,
            },
        )

    plt.tight_layout()
    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


def save_cross_host_latency_chart(
    case_summary,
    mean_column,
    sd_column,
    title,
    path,
    formula,
):
    """
    Plot a cross-host latency in seconds for readability.

    The underlying CSV values remain in milliseconds.
    Error bars show SD across the three independent runs.
    """
    data = deployment_order(case_summary)

    means_seconds = data[mean_column].to_numpy() / 1000.0
    sd_seconds = data[sd_column].fillna(0).to_numpy() / 1000.0

    figure_height = max(7, len(data) * 0.65)
    fig, ax = plt.subplots(figsize=(15, figure_height))

    bars = ax.barh(
        data["case_label"],
        means_seconds,
        xerr=sd_seconds,
        capsize=4,
    )

    ax.invert_yaxis()
    ax.set_xlabel("Latency (seconds)")
    ax.set_ylabel("Backend combination")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)

    add_value_labels(
        ax,
        bars,
        means_seconds,
        sd_seconds,
        "s",
    )

    largest = max(means_seconds + sd_seconds)
    ax.set_xlim(0, largest * 1.30)

    ax.text(
        0.98,
        0.98,
        f"{formula}\nApproximate cross-host metric\nAffected by NTP clock-offset uncertainty",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "alpha": 0.85,
        },
    )

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def add_grouped_bar_labels(ax, bars, values, suffix):
    """Write values at the end of grouped horizontal bars."""
    largest_value = max(values) if len(values) else 1
    label_gap = largest_value * 0.012

    for bar, value in zip(bars, values):
        ax.text(
            value + label_gap,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f} {suffix}",
            va="center",
            fontsize=8,
        )


def save_producer_gateway_rate_chart(case_summary, path):
    """
    Compare the rate offered by Producer with the rate observed
    at the Gateway ingress.

    Producer offered rate:
        N * 1e9 / (T1_last - T1_first)

    Gateway ingress rate:
        N * 1e9 / (T2_last - T2_first)
    """
    data = deployment_order(case_summary)

    producer_rate = data[
        "producer_offered_rate_msg_s_mean"
    ].to_numpy()

    gateway_ingress_rate = data[
        "gateway_ingress_rate_msg_s_mean"
    ].to_numpy()

    positions = np.arange(len(data))
    bar_height = 0.34

    figure_height = max(7, len(data) * 0.70)
    fig, ax = plt.subplots(figsize=(15, figure_height))

    producer_bars = ax.barh(
        positions - bar_height / 2,
        producer_rate,
        bar_height,
        label="Producer offered rate",
    )

    ingress_bars = ax.barh(
        positions + bar_height / 2,
        gateway_ingress_rate,
        bar_height,
        label="Gateway ingress rate",
    )

    ax.set_yticks(positions, data["case_label"])
    ax.invert_yaxis()
    ax.set_xlabel("Messages per second")
    ax.set_ylabel("Backend combination")
    ax.set_title("Producer Offered Rate versus Gateway Ingress Rate")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()

    add_grouped_bar_labels(
        ax,
        producer_bars,
        producer_rate,
        "msg/s",
    )
    add_grouped_bar_labels(
        ax,
        ingress_bars,
        gateway_ingress_rate,
        "msg/s",
    )

    largest_value = max(
        producer_rate.max(),
        gateway_ingress_rate.max(),
    )
    ax.set_xlim(0, largest_value * 1.18)

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def save_gateway_consumer_rate_chart(case_summary, path):
    """
    Compare Gateway egress rate with Consumer delivered throughput.

    Gateway egress rate:
        N * 1e9 / (T3_last - T3_first)

    Consumer delivered throughput:
        N * 1e9 / (T4_last - T4_first)

    Both rates are calculated independently on one host, so they do not
    require inter-host clock synchronization.
    """
    data = deployment_order(case_summary)

    gateway_egress_rate = data[
        "gateway_egress_rate_msg_s_mean"
    ].to_numpy()

    consumer_rate = data[
        "consumer_throughput_msg_s_mean"
    ].to_numpy()

    positions = np.arange(len(data))
    bar_height = 0.34

    figure_height = max(7, len(data) * 0.70)
    fig, ax = plt.subplots(figsize=(15, figure_height))

    egress_bars = ax.barh(
        positions - bar_height / 2,
        gateway_egress_rate,
        bar_height,
        label="Gateway egress rate",
    )

    consumer_bars = ax.barh(
        positions + bar_height / 2,
        consumer_rate,
        bar_height,
        label="Consumer delivered throughput",
    )

    ax.set_yticks(positions, data["case_label"])
    ax.invert_yaxis()
    ax.set_xlabel("Messages per second")
    ax.set_ylabel("Backend combination")
    ax.set_title("Gateway Egress Rate versus Consumer Delivered Throughput")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()

    add_grouped_bar_labels(
        ax,
        egress_bars,
        gateway_egress_rate,
        "msg/s",
    )
    add_grouped_bar_labels(
        ax,
        consumer_bars,
        consumer_rate,
        "msg/s",
    )

    largest_value = max(
        gateway_egress_rate.max(),
        consumer_rate.max(),
    )
    ax.set_xlim(0, largest_value * 1.18)

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def save_throughput_latency_scatter(case_summary, path):
    """
    Show the relationship between Gateway latency and Consumer throughput.
    """
    data = deployment_order(case_summary)

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    ax.scatter(
        data["gateway_latency_ms_mean_mean"],
        data["consumer_throughput_msg_s_mean"],
    )

    for _, row in data.iterrows():
        ax.annotate(
            short_case_name(row["case_id"]),
            (
                row["gateway_latency_ms_mean_mean"],
                row["consumer_throughput_msg_s_mean"],
            ),
            xytext=(5, 5),
            textcoords="offset points",
        )

    ax.set_xlabel(
        "Gateway mean latency, T3 - T2 (ms)"
    )
    ax.set_ylabel(
        "Consumer delivered throughput (msg/s)"
    )
    ax.set_title(
        "Throughput-Latency Trade-off"
    )
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gateway-results",
        required=True,
        help="Path containing final_case_*_deployment_run_* Gateway folders",
    )

    parser.add_argument(
        "--producer-results",
        required=True,
        help="Path containing matching Producer result folders",
    )

    parser.add_argument(
        "--consumer-results",
        required=True,
        help="Path containing matching Consumer result folders",
    )

    parser.add_argument(
        "--output",
        default="analysis_output/deployment",
        help="Output directory",
    )

    args = parser.parse_args()

    gateway_root = Path(
        args.gateway_results
    )
    producer_root = Path(
        args.producer_results
    )
    consumer_root = Path(
        args.consumer_results
    )

    output_dir = Path(
        args.output
    )
    figures_dir = output_dir / "figures"

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_rows = []

    gateway_runs = sorted(
        gateway_root.glob(
            "final_case_*_deployment_run_*"
        )
    )

    for gateway_run in gateway_runs:
        if "_failed_" in gateway_run.name:
            continue

        experiment_id = gateway_run.name
        producer_run = producer_root / experiment_id
        consumer_run = consumer_root / experiment_id

        run_rows.append(
            summarize_run(
                gateway_run,
                producer_run,
                consumer_run,
            )
        )

    run_summary = pd.DataFrame(
        run_rows
    ).sort_values(
        ["case_id", "run_number"]
    )

    case_summary = create_case_summary(
        run_summary
    )

    numeric_ranking = create_numeric_ranking(
        case_summary
    )

    professor_table = create_professor_table(
        numeric_ranking
    )

    detailed_table = create_detailed_table(
        case_summary
    )

    run_summary.to_csv(
        output_dir / "deployment_run_summary.csv",
        index=False,
        encoding=CSV_ENCODING,
    )

    case_summary.to_csv(
        output_dir / "deployment_case_summary.csv",
        index=False,
        encoding=CSV_ENCODING,
    )

    professor_table.to_csv(
        output_dir / "deployment_professor_table.csv",
        index=False,
        encoding=CSV_ENCODING,
    )

    numeric_ranking.to_csv(
        output_dir / "deployment_ranking.csv",
        index=False,
        encoding=CSV_ENCODING,
    )

    detailed_table.to_csv(
        output_dir / "deployment_detailed_table.csv",
        index=False,
        encoding=CSV_ENCODING,
    )

    highest_throughput = numeric_ranking.iloc[0]

    lowest_gateway_latency = numeric_ranking.loc[
        numeric_ranking[
            "Gateway latency mean (ms)"
        ].idxmin()
    ]

    lowest_cpu = numeric_ranking.loc[
        numeric_ranking[
            "CPU mean (%)"
        ].idxmin()
    ]

    negative_e2e_total = int(
        run_summary[
            "negative_approx_e2e_values"
        ].sum()
    )

    findings = (
        f"Highest Consumer throughput: "
        f"{highest_throughput['Case']} | "
        f"{highest_throughput['Analytics']} | "
        f"{highest_throughput['AI']} | "
        f"{highest_throughput['Crypto']} | "
        f"{highest_throughput['Consumer throughput mean (msg/s)']} msg/s\n"

        f"Gateway egress rate of the highest-throughput combination: "
        f"{highest_throughput['Gateway egress rate (msg/s)']} msg/s\n"

        f"Consumer delivered throughput of the highest-throughput combination: "
        f"{highest_throughput['Consumer throughput mean (msg/s)']} msg/s\n"

        f"Lowest Gateway latency: "
        f"{lowest_gateway_latency['Case']} | "
        f"{lowest_gateway_latency['Gateway latency mean (ms)']} ms\n"

        f"Lowest mean CPU: "
        f"{lowest_cpu['Case']} | "
        f"{lowest_cpu['CPU mean (%)']}%\n"

        f"Negative approximate E2E values across all runs: "
        f"{negative_e2e_total}\n"

        "Approximate T4-T1 latency is reported but is not used for "
        "primary ranking because it is affected by cross-host NTP "
        "clock-offset uncertainty.\n"
    )

    (
        output_dir / "deployment_key_findings.txt"
    ).write_text(
        findings,
        encoding="utf-8",
    )

    save_horizontal_bar_chart(
        case_summary,
        "consumer_throughput_msg_s_mean",
        "consumer_throughput_msg_s_std",
        "Consumer delivered throughput (msg/s)",
        "Consumer Delivered Throughput by Backend Combination",
        figures_dir / "01_consumer_throughput.png",
        "msg/s",
    )

    save_producer_gateway_rate_chart(
        case_summary,
        figures_dir / "02_producer_vs_gateway_ingress.png",
    )

    save_gateway_consumer_rate_chart(
        case_summary,
        figures_dir / "03_gateway_egress_vs_consumer.png",
    )

    save_horizontal_bar_chart(
        case_summary,
        "gateway_latency_ms_mean_mean",
        "gateway_latency_ms_mean_std",
        "Gateway latency, T3 - T2 (ms)",
        "Mean Gateway Boundary Latency",
        figures_dir / "04_gateway_latency_mean.png",
        "ms",
    )

    save_horizontal_bar_chart(
        case_summary,
        "gateway_latency_ms_p95_mean",
        "gateway_latency_ms_p95_std",
        "Gateway p95 latency, T3 - T2 (ms)",
        "P95 Gateway Boundary Latency",
        figures_dir / "05_gateway_latency_p95.png",
        "ms",
    )

    save_cross_host_latency_chart(
        case_summary,
        "producer_to_gateway_ms_mean_mean",
        "producer_to_gateway_ms_mean_std",
        "Approximate Mean Producer-to-Gateway Latency",
        figures_dir / "06_producer_to_gateway_mean.png",
        "T2 - T1",
    )

    save_cross_host_latency_chart(
        case_summary,
        "producer_to_gateway_ms_p95_mean",
        "producer_to_gateway_ms_p95_std",
        "Approximate P95 Producer-to-Gateway Latency",
        figures_dir / "07_producer_to_gateway_p95.png",
        "T2 - T1",
    )

    save_cross_host_latency_chart(
        case_summary,
        "gateway_to_consumer_ms_mean_mean",
        "gateway_to_consumer_ms_mean_std",
        "Approximate Mean Gateway-to-Consumer Latency",
        figures_dir / "08_gateway_to_consumer_mean.png",
        "T4 - T3",
    )

    save_cross_host_latency_chart(
        case_summary,
        "gateway_to_consumer_ms_p95_mean",
        "gateway_to_consumer_ms_p95_std",
        "Approximate P95 Gateway-to-Consumer Latency",
        figures_dir / "09_gateway_to_consumer_p95.png",
        "T4 - T3",
    )

    save_cross_host_latency_chart(
        case_summary,
        "approx_e2e_ms_mean_mean",
        "approx_e2e_ms_mean_std",
        "Approximate Mean End-to-End Latency",
        figures_dir / "10_approx_e2e_mean.png",
        "T4 - T1",
    )

    save_cross_host_latency_chart(
        case_summary,
        "approx_e2e_ms_p95_mean",
        "approx_e2e_ms_p95_std",
        "Approximate P95 End-to-End Latency",
        figures_dir / "11_approx_e2e_p95.png",
        "T4 - T1",
    )

    save_throughput_latency_scatter(
        case_summary,
        figures_dir / "12_throughput_latency_tradeoff.png",
    )

    save_horizontal_bar_chart(
        case_summary,
        "cpu_mean_percent_mean",
        "cpu_mean_percent_std",
        "CPU utilization (%)",
        "Mean CPU Utilization",
        figures_dir / "13_cpu_mean.png",
        "%",
    )

    save_horizontal_bar_chart(
        case_summary,
        "ram_mean_mb_mean",
        "ram_mean_mb_std",
        "RAM usage (MB)",
        "Mean RAM Usage",
        figures_dir / "14_ram_mean.png",
        "MB",
    )

    print(
        f"Deployment runs analyzed: "
        f"{len(run_summary)}"
    )

    print(
        f"Deployment combinations: "
        f"{len(case_summary)}"
    )

    print(
        f"Output directory: "
        f"{output_dir.resolve()}"
    )

    print(
        "Note: T2 - T1, T4 - T3, and T4 - T1 are approximate "
        "cross-host latencies affected by NTP clock-offset uncertainty. "
        "T3 - T2 is measured entirely on the Gateway."
    )


if __name__ == "__main__":
    main()
