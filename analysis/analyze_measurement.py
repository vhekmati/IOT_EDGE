#!/usr/bin/env python3

"""
Analyze all final Measurement-mode runs.

Expected input structure:

gateway_results/
    final_case_01_measurement_run_01/
        runtime_experiment_config.json
        measurement_raw.csv
        resource_usage.csv
    ...

Main outputs:
    measurement_run_summary.csv       one row per run (36 rows)
    measurement_case_summary.csv      one row per combination (12 rows)
    measurement_professor_table.csv   concise, sorted presentation table
    measurement_ranking.csv           sorted numeric table
    measurement_detailed_table.csv    detailed stage statistics
    figures/*.png                     readable comparison charts, including zoomed boxplots

Only the steady-state messages are analyzed. With 50 warm-up messages,
this means seq_id 51 ... 1050.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NS_TO_MS = 1_000_000.0
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
    """Return a presentation-friendly backend name."""
    return BACKEND_NAMES.get(name, name)


def short_case_name(case_id):
    """Convert case_11 to C11."""
    return case_id.replace("case_", "C").upper()


def add_case_labels(data):
    """
    Add a label that explains the complete backend combination.

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


def read_config(run_dir):
    with (run_dir / "runtime_experiment_config.json").open(
        "r", encoding="utf-8"
    ) as file:
        return json.load(file)


def add_latency_columns(data):
    """Convert timestamp pairs to latency values in milliseconds."""

    data = data.copy()

    # Stage latency formula:
    # stage_latency_ms = (stage_end_ns - stage_start_ns) / 1,000,000
    data["decrypt_ms"] = (
        data["decrypt_end_ns"] - data["decrypt_start_ns"]
    ) / NS_TO_MS
    data["analytics_ms"] = (
        data["analytics_end_ns"] - data["analytics_start_ns"]
    ) / NS_TO_MS
    data["ai_ms"] = (
        data["ai_end_ns"] - data["ai_start_ns"]
    ) / NS_TO_MS
    data["encrypt_ms"] = (
        data["encrypt_end_ns"] - data["encrypt_start_ns"]
    ) / NS_TO_MS

    # Total pipeline latency formula:
    # total_ms = (total_end_ns - total_start_ns) / 1,000,000
    data["total_ms"] = (
        data["total_end_ns"] - data["total_start_ns"]
    ) / NS_TO_MS

    # Crypto latency formula:
    # crypto_ms = decrypt_ms + encrypt_ms
    data["crypto_ms"] = data["decrypt_ms"] + data["encrypt_ms"]

    # Unattributed latency formula:
    # unattributed_ms = total_ms - sum(measured stage latencies)
    #
    # It includes JSON/base64 conversion, object creation, os.urandom(),
    # loop overhead, and other work outside the four stage timers.
    data["unattributed_ms"] = data["total_ms"] - (
        data["decrypt_ms"]
        + data["analytics_ms"]
        + data["ai_ms"]
        + data["encrypt_ms"]
    )

    return data


def summarize_resource_usage(run_dir):
    resource = pd.read_csv(run_dir / "resource_usage.csv")

    return {
        "cpu_mean_percent": resource["cpu_percent_total"].mean(),
        "cpu_p95_percent": percentile_95(resource["cpu_percent_total"]),
        "cpu_max_percent": resource["cpu_percent_total"].max(),
        "ram_mean_mb": resource["rss_mb_total"].mean(),
        "ram_max_mb": resource["rss_mb_total"].max(),
    }


def summarize_run(run_dir):
    config = read_config(run_dir)
    warmup_messages = int(config["warmup_messages"])

    raw = pd.read_csv(run_dir / "measurement_raw.csv")
    raw["seq_id"] = raw["seq_id"].astype(int)

    # Remove warm-up messages before calculating the KPI values.
    steady = raw[raw["seq_id"] > warmup_messages].copy()
    steady = add_latency_columns(steady)

    summary = {
        "experiment_id": config["experiment_id"],
        "case_id": config["combination_id"],
        "run_number": int(config["run_number"]),
        "analytics_backend": config["analytics_backend"],
        "ai_backend": config["ai_backend"],
        "crypto_backend": config["crypto_backend"],
        "steady_messages": len(steady),
    }

    latency_columns = [
        "decrypt_ms",
        "analytics_ms",
        "ai_ms",
        "encrypt_ms",
        "crypto_ms",
        "unattributed_ms",
        "total_ms",
    ]

    for column in latency_columns:
        summary[f"{column}_mean"] = steady[column].mean()
        summary[f"{column}_median"] = steady[column].median()
        summary[f"{column}_p95"] = percentile_95(steady[column])
        summary[f"{column}_p99"] = percentile_99(steady[column])

    summary.update(summarize_resource_usage(run_dir))

    # Per-message values are used only for descriptive boxplots.
    # They are not treated as independent ANOVA observations.
    steady["case_id"] = config["combination_id"]
    steady["run_number"] = int(config["run_number"])

    return summary, steady


def create_case_summary(run_summary):
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

    # Each combination has three independent runs.
    #
    # For every KPI:
    # case_mean = mean(KPI_run_1, KPI_run_2, KPI_run_3)
    # case_SD   = sample SD(KPI_run_1, KPI_run_2, KPI_run_3)
    case_summary = run_summary.groupby(group_columns)[numeric_columns].agg(
        ["mean", "std"]
    )

    case_summary.columns = [
        f"{metric}_{statistic}"
        for metric, statistic in case_summary.columns
    ]

    return case_summary.reset_index()


def create_numeric_ranking(case_summary):
    """
    Create the main numeric table, sorted from lowest to highest latency.

    Mean and SD remain in separate numeric columns so Excel can still
    filter, sort, calculate, and plot them.
    """
    table = case_summary[
        [
            "case_id",
            "analytics_backend",
            "ai_backend",
            "crypto_backend",
            "total_ms_mean_mean",
            "total_ms_mean_std",
            "total_ms_p95_mean",
            "total_ms_p95_std",
            "analytics_ms_mean_mean",
            "analytics_ms_p95_mean",
            "ai_ms_mean_mean",
            "ai_ms_p95_mean",
            "crypto_ms_mean_mean",
            "crypto_ms_p95_mean",
            "cpu_mean_percent_mean",
            "ram_mean_mb_mean",
        ]
    ].copy()

    table.columns = [
        "Case",
        "Analytics",
        "AI",
        "Crypto",
        "Total latency mean (ms)",
        "Total latency SD (ms)",
        "Total p95 mean (ms)",
        "Total p95 SD (ms)",
        "Analytics mean (ms)",
        "Analytics p95 (ms)",
        "AI mean (ms)",
        "AI p95 (ms)",
        "Crypto mean (ms)",
        "Crypto p95 (ms)",
        "CPU mean (%)",
        "RAM mean (MB)",
    ]

    table["Analytics"] = table["Analytics"].map(readable_backend_name)
    table["AI"] = table["AI"].map(readable_backend_name)
    table["Crypto"] = table["Crypto"].map(readable_backend_name)
    table["Case"] = table["Case"].map(short_case_name)

    table = table.sort_values(
        "Total latency mean (ms)"
    ).reset_index(drop=True)
    table.insert(0, "Latency rank", range(1, len(table) + 1))

    numeric_columns = table.select_dtypes(include="number").columns
    table[numeric_columns] = table[numeric_columns].round(3)

    return table


def create_professor_table(numeric_ranking):
    """
    Create a concise presentation table.

    Total mean and total p95 are shown as mean ± SD.
    Stage mean and p95 values remain visible for Analytics, AI, and Crypto.
    """
    table = numeric_ranking.copy()

    table["Total mean ± SD (ms)"] = table.apply(
        lambda row: (
            f"{row['Total latency mean (ms)']:.3f} ± "
            f"{row['Total latency SD (ms)']:.3f}"
        ),
        axis=1,
    )
    table["Total p95 ± SD (ms)"] = table.apply(
        lambda row: (
            f"{row['Total p95 mean (ms)']:.3f} ± "
            f"{row['Total p95 SD (ms)']:.3f}"
        ),
        axis=1,
    )

    return table[
        [
            "Latency rank",
            "Case",
            "Analytics",
            "AI",
            "Crypto",
            "Total mean ± SD (ms)",
            "Total p95 ± SD (ms)",
            "Analytics mean (ms)",
            "Analytics p95 (ms)",
            "AI mean (ms)",
            "AI p95 (ms)",
            "Crypto mean (ms)",
            "Crypto p95 (ms)",
            "CPU mean (%)",
            "RAM mean (MB)",
        ]
    ]


def create_detailed_table(case_summary):
    """Create a detailed table for report writing and deeper analysis."""

    identity_columns = [
        "case_id",
        "analytics_backend",
        "ai_backend",
        "crypto_backend",
    ]

    metric_columns = []
    for stage in [
        "decrypt_ms",
        "analytics_ms",
        "ai_ms",
        "encrypt_ms",
        "crypto_ms",
        "unattributed_ms",
        "total_ms",
    ]:
        metric_columns.extend(
            [
                f"{stage}_mean_mean",
                f"{stage}_mean_std",
                f"{stage}_p95_mean",
                f"{stage}_p95_std",
                f"{stage}_p99_mean",
                f"{stage}_p99_std",
            ]
        )

    metric_columns.extend(
        [
            "cpu_mean_percent_mean",
            "cpu_mean_percent_std",
            "cpu_p95_percent_mean",
            "ram_mean_mb_mean",
            "ram_mean_mb_std",
            "ram_max_mb_mean",
        ]
    )

    table = case_summary[identity_columns + metric_columns].copy()
    table = table.sort_values("total_ms_mean_mean").reset_index(drop=True)
    table.insert(0, "latency_rank", range(1, len(table) + 1))

    numeric_columns = table.select_dtypes(include="number").columns
    table[numeric_columns] = table[numeric_columns].round(3)
    return table


def latency_order(case_summary):
    """Use one consistent order for all Measurement charts."""
    data = case_summary.sort_values("total_ms_mean_mean").reset_index(drop=True)
    return add_case_labels(data)


def add_value_labels(ax, bars, means, standard_deviations, suffix):
    """Write mean ± SD at the end of each horizontal bar."""
    largest_value = max(means) if len(means) else 1
    label_gap = largest_value * 0.015

    for bar, mean, sd in zip(bars, means, standard_deviations):
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
):
    """
    Save a readable horizontal bar chart.

    Cases use the same total-latency order in every Measurement chart.
    Error bars show SD across the three independent runs.
    """
    data = latency_order(case_summary)
    means = data[mean_column].to_numpy()
    standard_deviations = data[sd_column].fillna(0).to_numpy()

    figure_height = max(7, len(data) * 0.65)
    fig, ax = plt.subplots(figsize=(15, figure_height))

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

    add_value_labels(
        ax,
        bars,
        means,
        standard_deviations,
        suffix,
    )

    right_limit = max(means + standard_deviations) * 1.30
    ax.set_xlim(0, right_limit)

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def save_stage_breakdown(case_summary, path):
    """
    Show the mean contribution of each main processing component.

    Crypto combines Decrypt and Encrypt:
    crypto_ms = decrypt_ms + encrypt_ms
    """
    data = latency_order(case_summary)

    stages = [
        ("analytics_ms_mean_mean", "Analytics"),
        ("ai_ms_mean_mean", "AI"),
        ("crypto_ms_mean_mean", "Crypto"),
        ("unattributed_ms_mean_mean", "Unattributed"),
    ]

    figure_height = max(7, len(data) * 0.65)
    fig, ax = plt.subplots(figsize=(15, figure_height))
    left = np.zeros(len(data))

    for column, label in stages:
        values = data[column].to_numpy()
        ax.barh(
            data["case_label"],
            values,
            left=left,
            label=label,
        )
        left += values

    total_values = data["total_ms_mean_mean"].to_numpy()
    label_gap = max(total_values) * 0.015

    for index, total in enumerate(total_values):
        ax.text(
            total + label_gap,
            index,
            f"{total:.2f} ms",
            va="center",
            fontsize=8,
        )

    ax.invert_yaxis()
    ax.set_xlabel("Mean latency (ms)")
    ax.set_ylabel("Backend combination")
    ax.set_title("Measurement mode: pipeline latency breakdown")
    ax.legend()

    ax.set_xlim(0, max(total_values) * 1.18)

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def save_boxplot_for_cases(
    all_messages,
    ordered_cases,
    path,
    title,
):
    """
    Create one horizontal box plot for the selected combinations.

    The line inside each box is the median.
    The diamond marker is the arithmetic mean.
    Outliers are hidden to keep the figure clean.

    Message-level values from the three runs are pooled only for this
    descriptive figure. Statistical tests must use run-level summaries.
    """
    values = [
        all_messages.loc[
            all_messages["case_id"] == row["case_id"],
            "total_ms",
        ]
        for _, row in ordered_cases.iterrows()
    ]
    labels = ordered_cases["case_label"].tolist()

    figure_height = max(5, len(labels) * 0.65)
    fig, ax = plt.subplots(figsize=(15, figure_height))

    ax.boxplot(
        values,
        tick_labels=labels,
        vert=False,
        showmeans=True,
        showfliers=False,
        meanprops={
            "marker": "D",
            "markersize": 5,
        },
        medianprops={
            "linewidth": 1.5,
        },
    )

    ax.invert_yaxis()
    ax.set_xlabel("Total latency (ms)")
    ax.set_ylabel("Backend combination")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)

    # Short explanation for the visual markers.
    ax.text(
        0.98,
        0.98,
        "Orange line = median\nGreen diamond = mean\nOutliers hidden",
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


def save_boxplots(all_messages, case_summary, figures_dir):
    """
    Save one overall box plot plus two zoomed plots.

    The overall plot shows the large performance gap between EdgeTPU
    and TFLite CPU. The two zoomed plots make the smaller differences
    inside each AI-backend group easier to see.
    """
    ordered_cases = latency_order(case_summary)

    save_boxplot_for_cases(
        all_messages,
        ordered_cases,
        figures_dir / "04_total_latency_boxplot_all.png",
        "Total Latency Distribution by Backend Combination",
    )

    edgetpu_cases = ordered_cases[
        ordered_cases["ai_backend"] == "edgetpu"
    ].reset_index(drop=True)

    save_boxplot_for_cases(
        all_messages,
        edgetpu_cases,
        figures_dir / "04a_total_latency_boxplot_edgetpu.png",
        "Total Latency Distribution: EdgeTPU Combinations",
    )

    cpu_cases = ordered_cases[
        ordered_cases["ai_backend"] == "cpu"
    ].reset_index(drop=True)

    save_boxplot_for_cases(
        all_messages,
        cpu_cases,
        figures_dir / "04b_total_latency_boxplot_cpu.png",
        "Total Latency Distribution: TFLite CPU Combinations",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gateway-results",
        required=True,
        help="Path containing final_case_*_measurement_run_* folders",
    )
    parser.add_argument(
        "--output",
        default="analysis_output/measurement",
        help="Output directory",
    )
    args = parser.parse_args()

    gateway_results = Path(args.gateway_results)
    output_dir = Path(args.output)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    run_rows = []
    message_tables = []

    run_dirs = sorted(gateway_results.glob("final_case_*_measurement_run_*"))

    for run_dir in run_dirs:
        if "_failed_" in run_dir.name:
            continue

        summary, steady_messages = summarize_run(run_dir)
        run_rows.append(summary)
        message_tables.append(steady_messages)

    run_summary = pd.DataFrame(run_rows).sort_values(
        ["case_id", "run_number"]
    )
    case_summary = create_case_summary(run_summary)
    numeric_ranking = create_numeric_ranking(case_summary)
    professor_table = create_professor_table(numeric_ranking)
    detailed_table = create_detailed_table(case_summary)
    all_messages = pd.concat(message_tables, ignore_index=True)

    run_summary.to_csv(
        output_dir / "measurement_run_summary.csv",
        index=False,
        encoding=CSV_ENCODING,
    )
    case_summary.to_csv(
        output_dir / "measurement_case_summary.csv",
        index=False,
        encoding=CSV_ENCODING,
    )
    professor_table.to_csv(
        output_dir / "measurement_professor_table.csv",
        index=False,
        encoding=CSV_ENCODING,
    )
    numeric_ranking.to_csv(
        output_dir / "measurement_ranking.csv",
        index=False,
        encoding=CSV_ENCODING,
    )
    detailed_table.to_csv(
        output_dir / "measurement_detailed_table.csv",
        index=False,
        encoding=CSV_ENCODING,
    )

    fastest = numeric_ranking.iloc[0]
    lowest_cpu = numeric_ranking.loc[
        numeric_ranking["CPU mean (%)"].idxmin()
    ]
    lowest_ram = numeric_ranking.loc[
        numeric_ranking["RAM mean (MB)"].idxmin()
    ]

    findings = (
        f"Fastest combination: {fastest['Case']} | "
        f"{fastest['Analytics']} | {fastest['AI']} | {fastest['Crypto']} | "
        f"total latency = {fastest['Total latency mean (ms)']} ms\n"
        f"Lowest mean CPU: {lowest_cpu['Case']} | "
        f"{lowest_cpu['CPU mean (%)']}%\n"
        f"Lowest mean RAM: {lowest_ram['Case']} | "
        f"{lowest_ram['RAM mean (MB)']} MB\n"
    )
    (output_dir / "measurement_key_findings.txt").write_text(
        findings,
        encoding="utf-8",
    )

    save_horizontal_bar_chart(
        case_summary,
        "total_ms_mean_mean",
        "total_ms_mean_std",
        "Total latency (ms)",
        "Measurement mode: mean pipeline latency",
        figures_dir / "01_total_latency_mean.png",
        "ms",
    )
    save_horizontal_bar_chart(
        case_summary,
        "total_ms_p95_mean",
        "total_ms_p95_std",
        "p95 latency (ms)",
        "Measurement mode: p95 pipeline latency",
        figures_dir / "02_total_latency_p95.png",
        "ms",
    )
    save_stage_breakdown(
        case_summary,
        figures_dir / "03_stage_latency_breakdown.png",
    )
    save_boxplots(
        all_messages,
        case_summary,
        figures_dir,
    )
    save_horizontal_bar_chart(
        case_summary,
        "cpu_mean_percent_mean",
        "cpu_mean_percent_std",
        "CPU utilization (%)",
        "Measurement mode: mean CPU utilization",
        figures_dir / "05_cpu_mean.png",
        "%",
    )
    save_horizontal_bar_chart(
        case_summary,
        "ram_mean_mb_mean",
        "ram_mean_mb_std",
        "RAM usage (MB)",
        "Measurement mode: mean RAM usage",
        figures_dir / "06_ram_mean.png",
        "MB",
    )

    print(f"Measurement runs analyzed: {len(run_summary)}")
    print(f"Measurement combinations: {len(case_summary)}")
    print(f"Output directory: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
