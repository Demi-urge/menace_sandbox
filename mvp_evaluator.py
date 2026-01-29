"""Deterministic MVP ROI evaluator based on output streams.

This module implements a minimal, fully deterministic, and monotonic scoring
function. More stdout increases the score, while more stderr decreases it. The
formula relies only on stable transforms of simple counts (length and line
count), has no thresholds tied to specific messages, uses no learning or
external state, and remains JSON-serializable by bounding the result.
"""

from __future__ import annotations

import math

__all__ = ["evaluate_mvp_roi"]


def evaluate_mvp_roi(stdout: str, stderr: str) -> float:
    """Return a deterministic ROI score based on stdout and stderr volume.

    The score is monotonic with respect to basic success signals: increasing
    stdout (length or line count) never decreases the ROI, while increasing
    stderr never increases it. The computation is pure and deterministic, relies
    only on bounded transforms of counts, and avoids any ML, dynamic weights, or
    external state. Unexpected inputs are coerced to ``str`` safely, and any
    evaluation error returns the sentinel ``-1.0``. The return value is always a
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

        stdout_len = len(safe_stdout)
        stderr_len = len(safe_stderr)
        stdout_lines = safe_stdout.count("\n") + (1 if safe_stdout else 0)
        stderr_lines = safe_stderr.count("\n") + (1 if safe_stderr else 0)

        stdout_signal = (
            (1.0 if safe_stdout else 0.0)
            + math.log1p(stdout_len)
            + math.log1p(stdout_lines)
        )
        stderr_signal = math.log1p(stderr_len) + math.log1p(stderr_lines)

        raw_score = stdout_signal - stderr_signal
        score = math.tanh(raw_score)

        if not math.isfinite(score):
            return -1.0

        return max(-1.0, min(1.0, float(score)))
    except Exception:
        return -1.0
