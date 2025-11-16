from __future__ import annotations

"""Utilities for benchmarking workflows and logging the results."""

from typing import Callable, Mapping
from time import perf_counter, sleep
import argparse
import importlib
import json
import logging
import os

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
from .roi_tracker import ROITracker
from .composite_workflow_scorer import CompositeWorkflowScorer


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
    err: Exception | None = None
    try:
        result = func()
        success = bool(result)
    except Exception as exc:  # pragma: no cover - defensive
        err = exc
        success = False
        logging.exception("workflow %s crashed", name)
    duration = perf_counter() - start
    cpu_time = 0.0
    cpu_user = 0.0
    cpu_system = 0.0
    mem_delta = 0.0
    mem_peak = 0.0
    net_delta = 0.0
    disk_read = 0.0
    disk_write = 0.0
    if proc and cpu_start:
        end_times = proc.cpu_times()
        cpu_user = float(end_times.user - cpu_start.user)
        cpu_system = float(end_times.system - cpu_start.system)
        cpu_time = cpu_user + cpu_system
        mem_end = proc.memory_info().rss
        mem_delta = float(max(mem_end - mem_start, 0)) / (1024 * 1024)
        mem_usage = float(mem_end) / (1024 * 1024)
        try:
            mem_peak = float(getattr(proc.memory_info(), "peak_wset", mem_end)) / (
                1024 * 1024
            )
        except Exception:
            mem_peak = mem_usage
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
            metrics_db.log_eval(name, "crash", 1.0 if err else 0.0)
            metrics_db.log_eval(name, "cpu_time", cpu_time)
            metrics_db.log_eval(name, "cpu_user_time", cpu_user)
            metrics_db.log_eval(name, "cpu_system_time", cpu_system)
            metrics_db.log_eval(name, "cpu_percent", cpu_percent)
            metrics_db.log_eval(name, "memory_delta", mem_delta)
            metrics_db.log_eval(name, "memory_usage", mem_usage)
            metrics_db.log_eval(name, "memory_peak", mem_peak)
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
                lat_min = float(np.min(vals)) if vals else latency
                lat_max = float(np.max(vals)) if vals else latency
            else:
                if vals:
                    vals_sorted = sorted(vals)
                    mid = len(vals_sorted) // 2
                    if len(vals_sorted) % 2:
                        median = float(vals_sorted[mid])
                    else:
                        median = float((vals_sorted[mid - 1] + vals_sorted[mid]) / 2.0)
                    lat_min = float(vals_sorted[0])
                    lat_max = float(vals_sorted[-1])
                else:
                    median = latency
                    lat_min = latency
                    lat_max = latency
            metrics_db.log_eval(name, "latency_median", median)
            metrics_db.log_eval(name, "latency_min", lat_min)
            metrics_db.log_eval(name, "latency_max", lat_max)
            if me:
                me.workflow_latency_p95_gauge.labels(name).set(p95)
                me.workflow_latency_median_gauge.labels(name).set(median)
                me.workflow_latency_min_gauge.labels(name).set(lat_min)
                me.workflow_latency_max_gauge.labels(name).set(lat_max)
                me.workflow_duration_gauge.labels(name).set(duration)
                me.workflow_cpu_percent_gauge.labels(name).set(cpu_percent)
                me.workflow_memory_gauge.labels(name).set(mem_delta)
                me.workflow_memory_usage_gauge.labels(name).set(mem_usage)
                me.workflow_peak_memory_gauge.labels(name).set(mem_peak)
                me.workflow_cpu_time_gauge.labels(name).set(cpu_time)
                me.workflow_cpu_user_time_gauge.labels(name).set(cpu_user)
                me.workflow_cpu_system_time_gauge.labels(name).set(cpu_system)
                me.workflow_net_io_gauge.labels(name).set(net_delta)
                me.workflow_disk_read_gauge.labels(name).set(disk_read)
                me.workflow_disk_write_gauge.labels(name).set(disk_write)
                if err:
                    me.workflow_crash_gauge.labels(name).inc()
            break
        except Exception:
            logging.exception("metrics logging failed on attempt %s", i + 1)
            if i == 2:
                raise
            sleep(0.1)

    rec = PathwayRecord(
        actions=name,
        inputs="",
        outputs=str(result) if err is None else f"error: {err}",
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


def benchmark_registered_workflows(
    workflows: Mapping[str, Callable[[], bool]],
    env_presets: list[Mapping[str, object]] | None,
    metrics_db: MetricsDB,
    pathway_db: PathwayDB,
    tracker: ROITracker,
) -> None:
    """Benchmark *workflows* under each environment preset.

    The metrics for every run are logged via ``MetricsDB`` and scored using
    :class:`CompositeWorkflowScorer` which records ROI deltas in ``tracker``.
    """

    presets = env_presets or [{}]
    baseline: dict[str, list[tuple]] = {n: metrics_db.fetch_eval(n) for n in workflows}
    scorer = CompositeWorkflowScorer(tracker=tracker)
    for preset in presets:
        for name, func in workflows.items():

            def _run() -> bool:
                env_backup = os.environ.copy()
                os.environ.update({k: str(v) for k, v in preset.items()})
                try:
                    return benchmark_workflow(func, metrics_db, pathway_db, name)
                finally:
                    os.environ.clear()
                    os.environ.update(env_backup)

            scorer.score_workflow(name, {name: _run})

    # statistical significance tests for key metrics
    for name in workflows:
        prev_rows = baseline.get(name, [])
        all_rows = metrics_db.fetch_eval(name)
        new_rows = all_rows[len(prev_rows):]

        def _metric_values(rows: list[tuple], metric: str) -> list[float]:
            return [float(v) for _, m, v, _ in rows if m == metric]

        metrics_to_test = [
            "duration",
            "cpu_time",
            "cpu_user_time",
            "cpu_system_time",
            "memory_delta",
            "memory_usage",
            "memory_peak",
            "cpu_percent",
            "latency",
            "latency_p95",
            "latency_median",
            "latency_min",
            "latency_max",
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
            if me:
                gauge_map = {
                    "latency_p95": me.workflow_latency_p95_gauge,
                    "latency_median": me.workflow_latency_median_gauge,
                    "latency_min": me.workflow_latency_min_gauge,
                    "latency_max": me.workflow_latency_max_gauge,
                    "memory_peak": me.workflow_peak_memory_gauge,
                    "cpu_user_time": me.workflow_cpu_user_time_gauge,
                    "cpu_system_time": me.workflow_cpu_system_time_gauge,
                }
                gauge = gauge_map.get(metric)
                if gauge is not None:
                    gauge.labels(name).set(new_vals[-1] if new_vals else 0.0)


__all__ = ["benchmark_workflow", "benchmark_registered_workflows"]


def _load_callable(path: str | os.PathLike[str]) -> Callable[[], bool]:
    """Return a callable referenced by ``path``.

    ``path`` may be a string or :class:`os.PathLike` object using
    ``module:func`` or ``module.func`` format.
    """

    path_str = str(path)

    if ":" in path_str:
        mod_name, func_name = path_str.split(":", 1)
    else:
        mod_name, func_name = path_str.rsplit(".", 1)
    func = getattr(importlib.import_module(mod_name), func_name)
    if not callable(func):  # pragma: no cover - defensive
        raise TypeError(f"{path_str} is not callable")
    return func


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for benchmarking a workflow.

    The callable should return ``True`` on success. Metrics and ROI results are
    stored in the default local databases including ``roi_results.db``.
    """

    parser = argparse.ArgumentParser(description="Benchmark a workflow and log results")
    parser.add_argument("callable", help="Dotted path to the workflow callable")
    parser.add_argument("--workflow-id", required=True, help="Workflow identifier")
    parser.add_argument("--run-id", help="Optional run identifier")
    args = parser.parse_args(argv)

    func = _load_callable(args.callable)
    scorer = CompositeWorkflowScorer()
    run_id, result = scorer.score_workflow(
        args.workflow_id, {func.__name__: func}, run_id=args.run_id
    )
    print(json.dumps({"run_id": run_id, **result}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
