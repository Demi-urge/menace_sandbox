"""Statistical utilities for Menace."""

from __future__ import annotations

import math
from statistics import NormalDist

try:  # optional SciPy dependency
    from scipy.stats import norm
    _norm_ppf = norm.ppf
except Exception:  # pragma: no cover - fallback
    _norm_ppf = NormalDist().inv_cdf


def wilson_score_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Return Wilson score interval for binomial proportion.

    Parameters
    ----------
    successes:
        Number of successful outcomes.
    n:
        Total number of trials.
    confidence:
        Two-tailed confidence level, e.g. ``0.95`` for 95%.
    """
    if n <= 0:
        return 0.0, 1.0
    z = _norm_ppf(1 - (1 - confidence) / 2)
    phat = successes / n
    denom = 1 + z**2 / n
    centre = phat + z**2 / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n)
    lower = (centre - margin) / denom
    upper = (centre + margin) / denom
    return max(0.0, float(lower)), min(1.0, float(upper))


def wilson_lower_bound(successes: int, n: int, confidence: float = 0.95) -> float:
    """Convenience wrapper returning only the lower bound."""
    return wilson_score_interval(successes, n, confidence)[0]


__all__ = ["wilson_score_interval", "wilson_lower_bound"]

