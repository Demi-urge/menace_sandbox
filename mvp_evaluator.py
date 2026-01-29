"""Deterministic ROI evaluator for MVP outputs.

The scoring is strictly deterministic and monotonic: more stdout increases the ROI
signal, and more stderr decreases it. It uses stable, bounded transforms of basic
counts (string length and line count) so behavior remains consistent even for
huge outputs. There are no tunable thresholds, no dynamic weights, no ML, and no
stateful or I/O-driven behavior.
"""

from __future__ import annotations

import math

__all__ = ["evaluate_roi"]


def evaluate_roi(stdout: str, stderr: str) -> float:
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
        safe_stdout = stdout if isinstance(stdout, str) else str(stdout)
        safe_stderr = stderr if isinstance(stderr, str) else str(stderr)

        stdout_len = len(safe_stdout)
        stderr_len = len(safe_stderr)
        stdout_lines = safe_stdout.count("\n") + (1 if safe_stdout else 0)
        stderr_lines = safe_stderr.count("\n") + (1 if safe_stderr else 0)

        stdout_signal = math.log1p(stdout_len) + math.log1p(stdout_lines)
        stderr_signal = math.log1p(stderr_len) + math.log1p(stderr_lines)

        raw_score = stdout_signal - stderr_signal
        score = math.tanh(raw_score)

        if not math.isfinite(score):
            return -1.0

        return max(-1.0, min(1.0, float(score)))
    except Exception:
        return -1.0
