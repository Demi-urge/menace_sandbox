from __future__ import annotations

"""ROI tracking helpers for the self-improvement engine.

This module provides a small shim around :mod:`self_improvement.metrics` so
that callers have a dedicated and aptly named entry point for updating the
alignment baseline.  The indirection keeps the public surface focused and makes
future refactors easier.
"""

from .metrics import _update_alignment_baseline as update_alignment_baseline

__all__ = ["update_alignment_baseline"]
