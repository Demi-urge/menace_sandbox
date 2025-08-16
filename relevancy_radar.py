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
_RELEVANCY_FLAGS_FILE = _BASE_DIR / "sandbox_data" / "relevancy_flags.json"

# In-memory counter for module usage.
_module_usage_counter: Counter[str] = Counter()

# In-memory store for relevancy flags produced by :func:`evaluate_relevancy`.
_relevancy_flags: Dict[str, str] = {}


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


def evaluate_relevancy(module_map: dict, usage_stats: dict) -> dict:
    """Return relevancy flags for modules based on ``usage_stats``.

    Modules absent from ``usage_stats`` are treated as having zero usage.
    The current heuristics are intentionally simple:

    - ``retire``  – modules with no recorded usage.
    - ``compress`` – modules used fewer than ``_COMPRESS_THRESHOLD`` times.
    - ``replace`` – modules used fewer than ``_REPLACE_THRESHOLD`` times.

    Results are persisted to :data:`_RELEVANCY_FLAGS_FILE` and cached in
    memory for access via :func:`flagged_modules`.
    """

    _COMPRESS_THRESHOLD = 5
    _REPLACE_THRESHOLD = 20

    flags: Dict[str, str] = {}
    for mod in module_map:
        count = int(usage_stats.get(mod, 0))
        if count == 0:
            flags[mod] = "retire"
        elif count <= _COMPRESS_THRESHOLD:
            flags[mod] = "compress"
        elif count <= _REPLACE_THRESHOLD:
            flags[mod] = "replace"

    _relevancy_flags.clear()
    _relevancy_flags.update(flags)

    _RELEVANCY_FLAGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _RELEVANCY_FLAGS_FILE.open("w", encoding="utf-8") as fh:
        json.dump(flags, fh, indent=2, sort_keys=True)

    return flags


def flagged_modules() -> Dict[str, str]:
    """Return the current relevancy recommendations."""

    if not _relevancy_flags and _RELEVANCY_FLAGS_FILE.exists():
        try:
            with _RELEVANCY_FLAGS_FILE.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                _relevancy_flags.update({str(k): str(v) for k, v in data.items()})
        except json.JSONDecodeError:
            pass

    return dict(_relevancy_flags)


def _save_usage_counts() -> None:
    """Persist merged usage statistics to disk."""
    counts = load_usage_stats()
    _MODULE_USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _MODULE_USAGE_FILE.open("w", encoding="utf-8") as fh:
        json.dump(counts, fh, indent=2, sort_keys=True)


# Ensure counters are persisted when the interpreter exits.
atexit.register(_save_usage_counts)

__all__ = [
    "track_module_usage",
    "load_usage_stats",
    "evaluate_relevancy",
    "flagged_modules",
]
