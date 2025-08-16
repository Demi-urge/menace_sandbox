"""Utilities for tracking and persisting module usage statistics.

This module maintains an in-memory counter of module usage that can be
incremented via :func:`track_module_usage`. On interpreter shutdown the
collected statistics are merged with any existing counts stored on disk
and persisted to ``sandbox_data/module_usage.json`` for later analysis.

The :func:`load_usage_stats` helper can be used to retrieve the
persisted counts merged with any data accumulated during the current
session.
"""

from __future__ import annotations

import atexit
import json
from collections import Counter
from pathlib import Path
from typing import Dict

# Path to the persistent usage statistics file.
_BASE_DIR = Path(__file__).resolve().parent
_MODULE_USAGE_FILE = _BASE_DIR / "sandbox_data" / "module_usage.json"

# In-memory counter for module usage.
_module_usage_counter: Counter[str] = Counter()


def track_module_usage(module: str) -> None:
    """Record usage of ``module``.

    Parameters
    ----------
    module:
        Name of the module to record.
    """

    _module_usage_counter[module] += 1


def load_usage_stats() -> Dict[str, int]:
    """Return usage statistics merged with any persisted counts."""
    counts: Counter[str] = Counter()

    if _MODULE_USAGE_FILE.exists():
        try:
            with _MODULE_USAGE_FILE.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                counts.update({str(k): int(v) for k, v in data.items()})
        except json.JSONDecodeError:
            # If the file is corrupt we ignore it to avoid raising during
            # shutdown or normal operation.
            pass

    counts.update(_module_usage_counter)
    return dict(counts)


def _save_usage_counts() -> None:
    """Persist merged usage statistics to disk."""
    counts = load_usage_stats()
    _MODULE_USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _MODULE_USAGE_FILE.open("w", encoding="utf-8") as fh:
        json.dump(counts, fh, indent=2, sort_keys=True)


# Ensure counters are persisted when the interpreter exits.
atexit.register(_save_usage_counts)

__all__ = ["track_module_usage", "load_usage_stats"]
