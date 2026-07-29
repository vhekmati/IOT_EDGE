#!/usr/bin/env python3

import argparse
import csv
import os
import signal
import time
from pathlib import Path


STOP = False
CLOCK_TICKS = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")


def handle_signal(signum, frame):
    del signum, frame
    global STOP
    STOP = True


def read_children(pid):
    children_path = Path(f"/proc/{pid}/task/{pid}/children")
    try:
        content = children_path.read_text(encoding="utf-8").strip()
    except OSError:
        return []
    if not content:
        return []
    return [int(value) for value in content.split()]


def process_tree(root_pid):
    result = []
    pending = [root_pid]
    seen = set()

    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        if not Path(f"/proc/{pid}").exists():
            continue
        result.append(pid)
        pending.extend(read_children(pid))

    return result


def read_process_stats(pid):
    stat_path = Path(f"/proc/{pid}/stat")
    statm_path = Path(f"/proc/{pid}/statm")
    status_path = Path(f"/proc/{pid}/status")

    stat_content = stat_path.read_text(encoding="utf-8")
    closing_parenthesis = stat_content.rfind(")")
    stat_values = stat_content[closing_parenthesis + 2 :].split()
    user_ticks = int(stat_values[11])
    system_ticks = int(stat_values[12])

    statm_values = statm_path.read_text(encoding="utf-8").split()
    vms_bytes = int(statm_values[0]) * PAGE_SIZE
    rss_bytes = int(statm_values[1]) * PAGE_SIZE

    threads = 0
    for line in status_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Threads:"):
            threads = int(line.split()[1])
            break

    return user_ticks + system_ticks, rss_bytes, vms_bytes, threads


def read_tree_usage(root_pid):
    total_ticks = 0
    total_rss = 0
    total_vms = 0
    total_threads = 0
    valid_processes = 0

    for pid in process_tree(root_pid):
        try:
            ticks, rss, vms, threads = read_process_stats(pid)
        except (OSError, ValueError, IndexError):
            continue
        total_ticks += ticks
        total_rss += rss
        total_vms += vms
        total_threads += threads
        valid_processes += 1

    return (
        total_ticks,
        total_rss,
        total_vms,
        total_threads,
        valid_processes,
    )


def atomic_write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.writer(output_file)
        writer.writerow([
            "experiment_id",
            "sample_index",
            "sample_time_ns",
            "cpu_percent_total",
            "rss_mb_total",
            "vms_mb_total",
            "num_threads_total",
            "num_processes",
        ])
        writer.writerows(rows)

    os.replace(temporary_path, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--output", required=True)
    parser.add_argument("--experiment-id", required=True)
    args = parser.parse_args()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    rows = []
    sample_index = 0
    previous_ticks, _, _, _, _ = read_tree_usage(args.pid)
    previous_time = time.monotonic()

    while not STOP and Path(f"/proc/{args.pid}").exists():
        time.sleep(args.interval)
        if STOP:
            break

        current_time = time.monotonic()
        try:
            ticks, rss, vms, threads, process_count = read_tree_usage(
                args.pid
            )
        except OSError:
            break

        elapsed_seconds = current_time - previous_time
        tick_delta = max(ticks - previous_ticks, 0)
        cpu_percent = (
            tick_delta / CLOCK_TICKS / elapsed_seconds * 100.0
            if elapsed_seconds > 0
            else 0.0
        )

        sample_index += 1
        rows.append([
            args.experiment_id,
            sample_index,
            time.time_ns(),
            round(cpu_percent, 2),
            round(rss / 1024 / 1024, 2),
            round(vms / 1024 / 1024, 2),
            threads,
            process_count,
        ])

        previous_ticks = ticks
        previous_time = current_time

    atomic_write(args.output, rows)


if __name__ == "__main__":
    main()
