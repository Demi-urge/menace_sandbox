from __future__ import annotations

"""Utilities for benchmarking workflows and logging the results."""

from typing import Callable
from time import perf_counter, sleep
import logging

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    np = None  # type: ignore

try:
    from . import metrics_exporter as me
except Exception:  # pragma: no cover - optional dependency
    me = None  # type: ignore

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore

from .data_bot import MetricsDB
from .neuroplasticity import PathwayDB, PathwayRecord, Outcome


def benchmark_workflow(
    func: Callable[[], bool],
    metrics_db: MetricsDB,
    pathway_db: PathwayDB,
    name: str = "workflow",
) -> bool:
    """Execute *func* and store evaluation metrics.

    Returns True if the workflow succeeded.
    """
    start = perf_counter()
    proc = psutil.Process() if psutil else None
    cpu_start = proc.cpu_times() if proc else None
    mem_start = proc.memory_info().rss if proc else 0
    mem_usage = mem_start
    net_start = psutil.net_io_counters() if psutil else None
    disk_start = psutil.disk_io_counters() if psutil else None
    success = False
    result = None
    try:
        result = func()
        success = bool(result)
    except Exception:
        success = False
    duration = perf_counter() - start
    cpu_time = 0.0
    mem_delta = 0.0
    net_delta = 0.0
    disk_read = 0.0
    disk_write = 0.0
    if proc and cpu_start:
        end_times = proc.cpu_times()
        cpu_time = float(
            (end_times.user + end_times.system) - (cpu_start.user + cpu_start.system)
        )
        mem_end = proc.memory_info().rss
        mem_delta = float(max(mem_end - mem_start, 0)) / (1024 * 1024)
        mem_usage = float(mem_end) / (1024 * 1024)
    if psutil and net_start:
        end_net = psutil.net_io_counters()
        net_delta = float(
            (end_net.bytes_sent + end_net.bytes_recv)
            - (net_start.bytes_sent + net_start.bytes_recv)
        )
    if psutil and disk_start:
        end_disk = psutil.disk_io_counters()
        disk_read = float(end_disk.read_bytes - disk_start.read_bytes)
        disk_write = float(end_disk.write_bytes - disk_start.write_bytes)
    cpu_percent = float((cpu_time / duration) * 100.0) if duration else 0.0
    latency = duration
    for i in range(3):
        try:
            metrics_db.log_eval(name, "duration", duration)
            metrics_db.log_eval(name, "latency", latency)
            metrics_db.log_eval(name, "success", 1.0 if success else 0.0)
            metrics_db.log_eval(name, "cpu_time", cpu_time)
            metrics_db.log_eval(name, "cpu_percent", cpu_percent)
            metrics_db.log_eval(name, "memory_delta", mem_delta)
            metrics_db.log_eval(name, "memory_usage", mem_usage)
            metrics_db.log_eval(name, "net_io", net_delta)
            metrics_db.log_eval(name, "disk_read", disk_read)
            metrics_db.log_eval(name, "disk_write", disk_write)
            if np is not None:
                vals = [
                    float(v)
                    for _, m, v, _ in metrics_db.fetch_eval(name)
                    if m == "latency"
                ]
                if vals:
                    p95 = float(np.percentile(vals, 95))
                else:
                    p95 = latency
            else:
                vals = [
                    float(v)
                    for _, m, v, _ in metrics_db.fetch_eval(name)
                    if m == "latency"
                ]
                if vals:
                    idx = int(0.95 * (len(vals) - 1))
                    p95 = float(sorted(vals)[idx])
                else:
                    p95 = latency
            metrics_db.log_eval(name, "latency_p95", p95)
            if np is not None:
                median = float(np.median(vals)) if vals else latency
            else:
                if vals:
                    vals_sorted = sorted(vals)
                    mid = len(vals_sorted) // 2
                    if len(vals_sorted) % 2:
                        median = float(vals_sorted[mid])
                    else:
                        median = float((vals_sorted[mid - 1] + vals_sorted[mid]) / 2.0)
                else:
                    median = latency
            metrics_db.log_eval(name, "latency_median", median)
            if me:
                me.workflow_latency_p95_gauge.labels(name).set(p95)
                me.workflow_latency_median_gauge.labels(name).set(median)
                me.workflow_duration_gauge.labels(name).set(duration)
                me.workflow_cpu_percent_gauge.labels(name).set(cpu_percent)
                me.workflow_memory_gauge.labels(name).set(mem_delta)
                me.workflow_memory_usage_gauge.labels(name).set(mem_usage)
                me.workflow_cpu_time_gauge.labels(name).set(cpu_time)
                me.workflow_net_io_gauge.labels(name).set(net_delta)
                me.workflow_disk_read_gauge.labels(name).set(disk_read)
                me.workflow_disk_write_gauge.labels(name).set(disk_write)
            break
        except Exception:
            logging.exception("metrics logging failed on attempt %s", i + 1)
            if i == 2:
                raise
            sleep(0.1)

    rec = PathwayRecord(
        actions=name,
        inputs="",
        outputs=str(result),
        exec_time=duration,
        resources=(
            f"cpu_time={cpu_time:.4f},memory_delta={mem_delta:.2f}MB,"
            f"cpu_percent={cpu_percent:.2f},memory_usage={mem_usage:.2f}MB,"
            f"net_io={net_delta:.0f},disk_read={disk_read:.0f},"
            f"disk_write={disk_write:.0f}"
        ),
        outcome=Outcome.SUCCESS if success else Outcome.FAILURE,
        roi=1.0 if success else 0.0,
    )

    for i in range(3):
        try:
            pathway_db.log(rec)
            break
        except Exception:
            logging.exception("pathway logging failed on attempt %s", i + 1)
            if i == 2:
                raise
            sleep(0.1)
    return success


import os
from typing import Mapping
from .roi_tracker import ROITracker


def benchmark_registered_workflows(
    workflows: Mapping[str, Callable[[], bool]],
    env_presets: list[Mapping[str, object]] | None,
    metrics_db: MetricsDB,
    pathway_db: PathwayDB,
    tracker: ROITracker,
) -> None:
    """Benchmark *workflows* under each environment preset.

    The metrics for every run are logged via ``MetricsDB`` and recorded in
    ``tracker`` using :meth:`ROITracker.update`.
    """

    presets = env_presets or [{}]
    baseline: dict[str, list[tuple]] = {
        n: metrics_db.fetch_eval(n) for n in workflows
    }
    stop_all = False
    for preset in presets:
        for name, func in workflows.items():
            prev_rows = metrics_db.fetch_eval(name)

            def _run() -> bool:
                env_backup = os.environ.copy()
                os.environ.update({k: str(v) for k, v in preset.items()})
                try:
                    return func()
                finally:
                    os.environ.clear()
                    os.environ.update(env_backup)

            benchmark_workflow(_run, metrics_db, pathway_db, name=name)
            new_rows = metrics_db.fetch_eval(name)[len(prev_rows) :]
            metrics: dict[str, float] = {}
            success = 0.0
            prev_success_vals = [float(v) for _, m, v, _ in prev_rows if m == "success"]
            prev_success = prev_success_vals[-1] if prev_success_vals else 0.0
            prev_lat_vals = [float(v) for _, m, v, _ in prev_rows if m == "latency_median"]
            prev_lat = prev_lat_vals[-1] if prev_lat_vals else 0.0
            for _, metric, value, _ in new_rows:
                try:
                    val = float(value)
                except Exception:
                    val = 0.0
                metrics[metric] = val
                if metric == "success":
                    success = val

            _, _, stop = tracker.update(prev_success, success, modules=[name], metrics=metrics)
            # early stop when improvement is negligible or success rate stagnates
            if abs(success - prev_success) <= tracker.diminishing() and abs(metrics.get("latency_median", metrics.get("latency", 0.0)) - prev_lat) <= tracker.diminishing():
                stop = True
            if len(prev_success_vals) >= 2 and len([v for _, m, v, _ in new_rows if m == "success"]) >= 2:
                try:
                    from scipy import stats

                    old = prev_success_vals[-2:]
                    new = [float(v) for _, m, v, _ in new_rows if m == "success"]
                    _, p_succ = stats.ttest_ind(old, new, equal_var=False)
                except Exception:
                    p_succ = 1.0
                if p_succ > 0.5 and abs(success - prev_success) <= tracker.diminishing():
                    stop = True
            if stop:
                stop_all = True
                break

        if stop_all:
            break

    # statistical significance tests for key metrics
    for name in workflows:
        prev_rows = baseline.get(name, [])
        all_rows = metrics_db.fetch_eval(name)
        new_rows = all_rows[len(prev_rows) :]

        def _metric_values(rows: list[tuple], metric: str) -> list[float]:
            return [float(v) for _, m, v, _ in rows if m == metric]

        metrics_to_test = [
            "duration",
            "cpu_time",
            "memory_delta",
            "memory_usage",
            "cpu_percent",
            "latency",
            "latency_p95",
            "latency_median",
            "net_io",
            "disk_read",
            "disk_write",
            "success",
        ]

        for metric in metrics_to_test:
            prev_vals = _metric_values(prev_rows, metric)
            new_vals = _metric_values(new_rows, metric)
            if len(prev_vals) >= 2 and len(new_vals) >= 2:
                try:
                    from scipy import stats

                    _, p_val = stats.ttest_ind(prev_vals, new_vals, equal_var=False)
                except Exception:
                    p_val = 1.0
            else:
                p_val = 1.0
            metrics_db.log_eval(name, f"{metric}_pvalue", float(p_val))
            tracker.metrics_history.setdefault(f"{metric}_pvalue", []).append(float(p_val))
            if me and metric in ("latency_p95", "latency_median"):
                gauge = (
                    me.workflow_latency_p95_gauge
                    if metric == "latency_p95"
                    else me.workflow_latency_median_gauge
                )
                gauge.labels(name).set(new_vals[-1] if new_vals else 0.0)


__all__ = ["benchmark_workflow", "benchmark_registered_workflows"]
