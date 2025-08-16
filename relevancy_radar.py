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
from typing import Dict, Iterable

from metrics_exporter import update_relevancy_metrics

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
    Thresholds and module whitelists are sourced from
    :class:`~sandbox_settings.SandboxSettings` when available:

    - ``retire``  – modules with no recorded usage.
    - ``compress`` – modules used fewer than 25% of
      ``relevancy_threshold``.
    - ``replace`` – modules used fewer than ``relevancy_threshold`` times.

    Results are persisted to :data:`_RELEVANCY_FLAGS_FILE` and cached in
    memory for access via :func:`flagged_modules`.
    """

    try:  # Import lazily to avoid heavy settings import on module load
        from sandbox_settings import SandboxSettings

        settings = SandboxSettings()
        replace_threshold = int(settings.relevancy_threshold)
        compress_threshold = max(1, replace_threshold // 4)
        whitelist = set(settings.relevancy_whitelist)
    except Exception:  # pragma: no cover - fall back to defaults
        replace_threshold = 20
        compress_threshold = 5
        whitelist = set()

    flags: Dict[str, str] = {}
    for mod in module_map:
        if mod in whitelist:
            continue
        count = int(usage_stats.get(mod, 0))
        if count == 0:
            flags[mod] = "retire"
        elif count <= compress_threshold:
            flags[mod] = "compress"
        elif count <= replace_threshold:
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
    "RelevancyRadar",
]


class RelevancyRadar:
    """High level helpers for working with module relevancy data."""

    @staticmethod
    def flag_unused_modules(module_map: Iterable[str]) -> Dict[str, str]:
        """Evaluate module usage and persist flags and orphan list.

        Parameters
        ----------
        module_map:
            Iterable of module identifiers to evaluate.
        """

        usage_stats = load_usage_stats()
        flags = evaluate_relevancy(dict.fromkeys(module_map), usage_stats)
        update_relevancy_metrics(flags)

        orphan_file = _BASE_DIR / "sandbox_data" / "orphan_modules.json"
        existing: list[str] = []
        if orphan_file.exists():
            try:
                with orphan_file.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    existing = [str(m) for m in data]
            except json.JSONDecodeError:
                existing = []

        retired = sorted(m for m, status in flags.items() if status == "retire")
        merged = sorted({*existing, *retired})

        orphan_file.parent.mkdir(parents=True, exist_ok=True)
        with orphan_file.open("w", encoding="utf-8") as fh:
            json.dump(merged, fh, indent=2, sort_keys=True)

        return flags
