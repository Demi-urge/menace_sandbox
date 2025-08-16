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
import builtins

from metrics_exporter import update_relevancy_metrics

# Path to the persistent usage statistics file.
_BASE_DIR = Path(__file__).resolve().parent
_MODULE_USAGE_FILE = _BASE_DIR / "sandbox_data" / "module_usage.json"
_RELEVANCY_FLAGS_FILE = _BASE_DIR / "sandbox_data" / "relevancy_flags.json"
# File used by :class:`RelevancyRadar` to persist detailed usage metrics.
_RELEVANCY_METRICS_FILE = _BASE_DIR / "sandbox_data" / "relevancy_metrics.json"

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
    """Track module usage and evaluate relevancy based on collected metrics."""

    def __init__(self, metrics_file: Path | None = None) -> None:
        self.metrics_file = Path(metrics_file) if metrics_file else _RELEVANCY_METRICS_FILE
        self._metrics: Dict[str, Dict[str, int]] = self._load_metrics()
        self._install_import_hook()
        atexit.register(self._persist_metrics)

    # ------------------------------------------------------------------
    # Metrics persistence helpers
    # ------------------------------------------------------------------
    def _load_metrics(self) -> Dict[str, Dict[str, int]]:
        data: Dict[str, Dict[str, int]] = {}
        if self.metrics_file.exists():
            try:
                with self.metrics_file.open("r", encoding="utf-8") as fh:
                    raw = json.load(fh)
                if isinstance(raw, dict):
                    for mod, counts in raw.items():
                        if isinstance(counts, dict):
                            data[str(mod)] = {
                                "imports": int(counts.get("imports", 0)),
                                "executions": int(counts.get("executions", 0)),
                            }
            except json.JSONDecodeError:
                data = {}
        return data

    def _persist_metrics(self) -> None:
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        with self.metrics_file.open("w", encoding="utf-8") as fh:
            json.dump(self._metrics, fh, indent=2, sort_keys=True)

    def _install_import_hook(self) -> None:
        if getattr(builtins, "_relevancy_radar_original_import", None):
            return

        original_import = builtins.__import__
        radar = self

        def tracked_import(name, globals=None, locals=None, fromlist=(), level=0):
            root = name.split(".")[0]
            info = radar._metrics.setdefault(root, {"imports": 0, "executions": 0})
            info["imports"] += 1
            return original_import(name, globals, locals, fromlist, level)

        builtins._relevancy_radar_original_import = original_import
        builtins.__import__ = tracked_import

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def track_usage(self, module_name: str) -> None:
        """Record an execution event for ``module_name`` and persist metrics."""

        info = self._metrics.setdefault(module_name, {"imports": 0, "executions": 0})
        info["executions"] += 1
        self._persist_metrics()

    def evaluate_relevance(self, threshold: float) -> Dict[str, str]:
        """Return modules with scores below ``threshold``.

        Modules with zero score are tagged ``retire`` while modules below the
        threshold are tagged ``compress``.
        """

        results: Dict[str, str] = {}
        for mod, counts in self._metrics.items():
            score = int(counts.get("imports", 0)) + int(counts.get("executions", 0))
            if score == 0:
                results[mod] = "retire"
            elif score < threshold:
                results[mod] = "compress"

        self._persist_metrics()
        return results

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
