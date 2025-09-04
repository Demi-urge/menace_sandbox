"""Deprecated sandbox results logger wrapper.

This module previously handled persistence of sandbox run metrics.  It now
forwards all calls to :mod:`sandbox_runner.scoring` so that legacy imports keep
working while metrics flow through the unified JSONL/summary pipeline.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict

from .sandbox_runner.scoring import record_run as _record_run


def record_run(metrics: Dict[str, Any]) -> None:
    """Forward *metrics* to :func:`sandbox_runner.scoring.record_run`.

    Parameters
    ----------
    metrics:
        Mapping containing runtime information. Keys such as ``success``,
        ``runtime`` and ``error`` are translated for the new API. Additional
        keys are passed through unchanged.
    """

    _record_run(
        SimpleNamespace(
            success=metrics.get("success"),
            duration=metrics.get("runtime"),
            failure=metrics.get("error"),
        ),
        metrics,
    )


__all__ = ["record_run"]

