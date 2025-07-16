from __future__ import annotations

"""Utilities for benchmarking workflows and logging the results."""

from typing import Callable
from time import perf_counter, sleep
import logging

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
    success = False
    result = None
    try:
        result = func()
        success = bool(result)
    except Exception:
        success = False
    duration = perf_counter() - start
    for i in range(3):
        try:
            metrics_db.log_eval(name, "duration", duration)
            metrics_db.log_eval(name, "success", 1.0 if success else 0.0)
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
        resources="",
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
            for _, metric, value, _ in new_rows:
                try:
                    val = float(value)
                except Exception:
                    val = 0.0
                metrics[metric] = val
                if metric == "success":
                    success = val

            tracker.update(0.0, success, modules=[name], metrics=metrics)


__all__ = ["benchmark_workflow", "benchmark_registered_workflows"]
