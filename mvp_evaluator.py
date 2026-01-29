"""Deterministic MVP ROI evaluator based on output stream sizes.

This module provides a fully deterministic, monotonic scoring function that
uses no stochasticity, no external state, and no ML. The score depends only on
generic lengths of stdout and stderr: more stdout length increases the score,
and more stderr length decreases it. The normalization is stable and bounded,
so the return value stays JSON-serializable without branching on specific
messages or tuned thresholds.
"""

from __future__ import annotations

import math

__all__ = ["evaluate_roi"]


def evaluate_roi(stdout: str, stderr: str) -> float:
    """Return a deterministic ROI score based on stdout and stderr volume.

    The score is monotonic with respect to basic success signals: increasing
    stdout length never decreases the ROI, while increasing stderr length never
    increases it. The computation is pure and deterministic, relies only on
    bounded transforms of sizes, and avoids any ML, dynamic weights, or external
    state. Unexpected inputs are coerced to ``str`` safely, and any evaluation
    error returns the sentinel ``-1.0``. The return value is always a
    JSON-serializable finite float.
    """
    try:
        def _coerce(value: object) -> str:
            if value is None:
                return ""
            if isinstance(value, bytes):
                return value.decode(errors="replace")
            return str(value)

        safe_stdout = _coerce(stdout)
        safe_stderr = _coerce(stderr)

        stdout_signal = math.log1p(len(safe_stdout))
        stderr_signal = math.log1p(len(safe_stderr))
        raw_score = stdout_signal - stderr_signal
        score = math.tanh(raw_score)

        if not math.isfinite(score):
            return -1.0

        return max(-1.0, min(1.0, float(score)))
    except Exception:
        return -1.0
