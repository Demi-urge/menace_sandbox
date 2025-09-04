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

__version__ = "1.0.0"

import atexit
import json
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
import os
from typing import Dict, Iterable, List, Callable, Any
import builtins

from db_router import DBRouter, GLOBAL_ROUTER, LOCAL_TABLES, init_db_router

import math

try:  # shared baseline tracker across modules
    from self_improvement.baseline_tracker import TRACKER as BASELINE_TRACKER
except Exception:  # pragma: no cover - fallback stub
    class _BaselineStub:
        def update(self, **metrics: float) -> None:
            pass

        def get(self, metric: str) -> float:
            return 0.0

        def std(self, metric: str) -> float:
            return 0.0

    BASELINE_TRACKER = _BaselineStub()

try:  # pragma: no cover - optional dependency
    import networkx as nx
except Exception:  # pragma: no cover - lightweight fallback
    from typing import Dict as _Dict, Set as _Set

    class _SimpleDiGraph:
        def __init__(self) -> None:
            self._adj: _Dict[str, _Set[str]] = defaultdict(set)

        def add_edge(self, u: str, v: str) -> None:
            self._adj[u].add(v)
            self._adj.setdefault(v, set())

        def __contains__(self, node: str) -> bool:  # pragma: no cover - trivial
            return node in self._adj

    def _descendants(graph: "_SimpleDiGraph", source: str) -> set[str]:
        seen: set[str] = set()
        stack = list(graph._adj.get(source, set()))
        while stack:
            node = stack.pop()
            if node not in seen:
                seen.add(node)
                stack.extend(graph._adj.get(node, set()))
        return seen

    class nx:  # type: ignore
        DiGraph = _SimpleDiGraph
        descendants = staticmethod(_descendants)

import inspect
import functools
import contextlib
from metrics_exporter import update_relevancy_metrics
from relevancy_metrics_db import RelevancyMetricsDB
try:  # pragma: no cover - allow running as script
    from .dynamic_path_router import resolve_path  # type: ignore
except Exception:  # pragma: no cover - fallback when executed directly
    from dynamic_path_router import resolve_path  # type: ignore

# Path to the persistent usage statistics file.
_SANDBOX_DATA_DIR = Path(resolve_path("sandbox_data"))
_MODULE_USAGE_FILE = _SANDBOX_DATA_DIR / "module_usage.json"
_RELEVANCY_FLAGS_FILE = _SANDBOX_DATA_DIR / "relevancy_flags.json"
# File used by :class:`RelevancyRadar` to persist detailed usage metrics.
_RELEVANCY_METRICS_FILE = _SANDBOX_DATA_DIR / "relevancy_metrics.json"
# Default location of the SQLite metrics store consumed by :func:`scan`.
_RELEVANCY_METRICS_DB = _SANDBOX_DATA_DIR / "relevancy_metrics.db"
# File used to persist the call graph between runs.
_RELEVANCY_CALL_GRAPH_FILE = _SANDBOX_DATA_DIR / "relevancy_call_graph.json"
# File used to persist the orphan module list between runs.
_ORPHAN_MODULES_FILE = _SANDBOX_DATA_DIR / "orphan_modules.json"

# In-memory counter for module usage storing timestamps for each call.
_module_usage_counter: Dict[str, List[int]] = defaultdict(list)

# In-memory store for relevancy flags produced by :func:`evaluate_relevancy`.
_relevancy_flags: Dict[str, str] = {}


def track_module_usage(module: str) -> None:
    """Record usage of ``module``.

    Parameters
    ----------
    module:
        Name of the module to record.
    """

    _module_usage_counter[module].append(int(time.time()))


def _relevancy_cutoff() -> float:
    """Return the timestamp cutoff for relevancy calculations."""
    try:
        days_env = int(os.getenv("RELEVANCY_WINDOW_DAYS", "0"))
        if days_env:
            return time.time() - days_env * 86400
    except Exception:
        pass
    try:  # pragma: no cover - optional dependency
        from sandbox_settings import SandboxSettings

        days = int(SandboxSettings().relevancy_window_days)
    except Exception:  # pragma: no cover - default if settings unavailable
        days = 30
    return time.time() - days * 86400


def _decay_factor(ts: int, *, now: float | None = None, window: float | None = None) -> float:
    """Return an exponential decay factor for ``ts`` relative to the relevancy window."""
    now = float(now if now is not None else time.time())
    if window is None:
        window = max(now - _relevancy_cutoff(), 1.0)
    age = max(0.0, now - float(ts))
    return math.exp(-age / window)


def _extract_impact(value: Any) -> float | None:
    """Best-effort extraction of an ROI/impact score from ``value``."""

    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for key in ("roi_delta", "impact", "roi"):
            val = value.get(key)
            if isinstance(val, (int, float)):
                return float(val)
    if isinstance(value, (list, tuple)) and value:
        for item in reversed(value):
            res = _extract_impact(item)
            if res is not None:
                return res
    for attr in ("roi_delta", "impact", "roi"):
        try:
            val = getattr(value, attr)
        except Exception:
            continue
        res = _extract_impact(val)
        if res is not None:
            return res
    return None


def load_usage_stats() -> Dict[str, float]:
    """Return usage statistics merged with any persisted counts using decay weighting."""
    cutoff = _relevancy_cutoff()
    now = time.time()
    window = max(now - cutoff, 1.0)
    merged: Dict[str, List[int]] = defaultdict(list)

    if _MODULE_USAGE_FILE.exists():
        try:
            with _MODULE_USAGE_FILE.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                for mod, timestamps in data.items():
                    if isinstance(timestamps, list):
                        merged[mod].extend(int(t) for t in timestamps)
        except json.JSONDecodeError:
            # Ignore corrupt files during normal operation.
            pass

    for mod, ts_list in _module_usage_counter.items():
        merged[mod].extend(ts_list)

    # Filter out entries older than the cutoff
    recent_counts: Dict[str, float] = {}
    for mod, ts_list in merged.items():
        recent = [ts for ts in ts_list if ts >= cutoff]
        if recent:
            recent_counts[mod] = sum(
                _decay_factor(ts, now=now, window=window) for ts in recent
            )
            merged[mod] = recent

    return recent_counts


def evaluate_relevancy(
    module_map: dict, usage_stats: dict, impact_stats: dict | None = None
) -> dict:
    """Return relevancy flags for modules based on usage and impact data.

    ``usage_stats`` may map module names either to raw timestamp lists or to
    pre-aggregated counts. Any entries older than ``relevancy_window_days`` are
    discarded before evaluation.

    ``impact_stats`` maps module names to cumulative impact scores such as ROI
    deltas. These scores are added to the usage counts when computing the final
    relevancy score for each module. Modules absent from either mapping are
    treated as having zero usage/impact.

    Thresholds are derived from per-module baselines tracked by
    :class:`~self_improvement.baseline_tracker.BaselineTracker` when
    available. For each module a moving average and standard deviation are
    computed and used to derive ``replace`` and ``compress`` thresholds:

    - ``retire``  – modules with no recorded score.
    - ``compress`` – score below ``avg - 2 * k * std``.
    - ``replace`` – score below ``avg - k * std``.

    ``k`` is the ``relevancy_deviation_multiplier`` setting (default ``1``).

    Module whitelists are honoured via
    :class:`~sandbox_settings.SandboxSettings` when available. Results are
    persisted to :data:`_RELEVANCY_FLAGS_FILE` and cached in memory for
    access via :func:`flagged_modules`.
    """

    cutoff = _relevancy_cutoff()
    now = time.time()
    window = max(now - cutoff, 1.0)
    counts: Dict[str, float] = {}
    for mod, data in usage_stats.items():
        if isinstance(data, list):
            counts[mod] = float(
                sum(_decay_factor(ts, now=now, window=window) for ts in data if ts >= cutoff)
            )
        else:
            counts[mod] = float(data)

    impact_stats = impact_stats or {}
    for mod, impact in impact_stats.items():
        counts[mod] = counts.get(mod, 0.0) + float(impact)

    usage_stats = counts

    try:  # Import lazily to avoid heavy settings import on module load
        from sandbox_settings import SandboxSettings

        settings = SandboxSettings()
        k = float(getattr(settings, "relevancy_deviation_multiplier", 1.0))
        whitelist = set(settings.relevancy_whitelist)
    except Exception:  # pragma: no cover - fall back to defaults
        k = 1.0
        whitelist = set()

    flags: Dict[str, str] = {}
    for mod in module_map:
        if mod in whitelist:
            continue
        score = float(usage_stats.get(mod, 0.0))
        metric_name = f"relevancy:{mod}"
        BASELINE_TRACKER.update(relevancy=score, **{metric_name: score})
        avg = BASELINE_TRACKER.get(metric_name)
        std = BASELINE_TRACKER.std(metric_name)
        replace_threshold = max(avg - k * std, 0.0)
        compress_threshold = max(avg - 2 * k * std, 0.0)
        if score == 0:
            flags[mod] = "retire"
        elif score < compress_threshold:
            flags[mod] = "compress"
        elif score < replace_threshold:
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
    """Persist merged usage statistics to disk including timestamps."""
    cutoff = _relevancy_cutoff()
    merged: Dict[str, List[int]] = defaultdict(list)

    if _MODULE_USAGE_FILE.exists():
        try:
            with _MODULE_USAGE_FILE.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                for mod, timestamps in data.items():
                    if isinstance(timestamps, list):
                        merged[mod].extend(int(t) for t in timestamps)
        except json.JSONDecodeError:
            pass

    for mod, ts_list in _module_usage_counter.items():
        merged[mod].extend(ts_list)

    # Drop entries older than the cutoff to avoid unbounded growth
    for mod, ts_list in list(merged.items()):
        recent = [ts for ts in ts_list if ts >= cutoff]
        if recent:
            merged[mod] = recent
        else:
            merged.pop(mod)

    _MODULE_USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _MODULE_USAGE_FILE.open("w", encoding="utf-8") as fh:
        json.dump({k: v for k, v in merged.items()}, fh, indent=2, sort_keys=True)


def call_graph_complexity() -> float:
    """Return a complexity score for the persisted relevancy call graph.

    The score is calculated as ``edges / nodes`` for the stored directed
    graph.  If the call graph file is missing or malformed ``0.0`` is
    returned.
    """

    if not _RELEVANCY_CALL_GRAPH_FILE.exists():
        return 0.0
    try:
        with _RELEVANCY_CALL_GRAPH_FILE.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except json.JSONDecodeError:
        return 0.0
    if not isinstance(raw, dict):
        return 0.0

    graph = nx.DiGraph()
    for caller, callees in raw.items():
        if isinstance(callees, list):
            for callee in callees:
                graph.add_edge(str(caller), str(callee))

    try:
        nodes = graph.number_of_nodes()
        edges = graph.number_of_edges()
    except AttributeError:  # pragma: no cover - fallback graph
        adj = getattr(graph, "_adj", {})
        nodes = len(adj)
        edges = sum(len(v) for v in adj.values())

    return float(edges) / nodes if nodes else 0.0


# Ensure counters are persisted when the interpreter exits.
atexit.register(_save_usage_counts)

__all__ = [
    "track_module_usage",
    "load_usage_stats",
    "evaluate_relevancy",
    "flagged_modules",
    "RelevancyRadar",
    "track_usage",
    "track",
    "record_output_impact",
    "evaluate_relevance",
    "evaluate_final_contribution",
    "radar",
    "call_graph_complexity",
]


class RelevancyRadar:
    """Track module usage and evaluate relevancy based on collected metrics."""

    def __init__(self, metrics_file: Path | None = None) -> None:
        self.metrics_file = Path(metrics_file) if metrics_file else _RELEVANCY_METRICS_FILE
        self._metrics: Dict[str, Dict[str, float | str]] = self._load_metrics()
        self._call_graph: Dict[str, set[str]] = self._load_call_graph()
        self._install_import_hook()
        atexit.register(self._persist_metrics)

    # ------------------------------------------------------------------
    # Metrics persistence helpers
    # ------------------------------------------------------------------
    def _load_metrics(self) -> Dict[str, Dict[str, float | str]]:
        data: Dict[str, Dict[str, float | str]] = {}
        if self.metrics_file.exists():
            try:
                with self.metrics_file.open("r", encoding="utf-8") as fh:
                    raw = json.load(fh)
                if isinstance(raw, dict):
                    for mod, counts in raw.items():
                        if isinstance(counts, dict):
                            entry: Dict[str, float | str] = {
                                "imports": float(counts.get("imports", 0.0)),
                                "executions": float(counts.get("executions", 0.0)),
                                "impact": float(counts.get("impact", 0.0)),
                                "output_impact": float(counts.get("output_impact", 0.0)),
                            }
                            annotation = counts.get("annotation")
                            if isinstance(annotation, str) and annotation:
                                entry["annotation"] = annotation
                            data[str(mod)] = entry
            except json.JSONDecodeError:
                data = {}
        return data

    def _load_call_graph(self) -> Dict[str, set[str]]:
        data: Dict[str, set[str]] = {}
        if _RELEVANCY_CALL_GRAPH_FILE.exists():
            try:
                with _RELEVANCY_CALL_GRAPH_FILE.open("r", encoding="utf-8") as fh:
                    raw = json.load(fh)
                if isinstance(raw, dict):
                    for caller, callees in raw.items():
                        if isinstance(callees, list):
                            data[str(caller)] = {str(c) for c in callees}
            except json.JSONDecodeError:
                data = {}
        return data

    def _persist_metrics(self) -> None:
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        with self.metrics_file.open("w", encoding="utf-8") as fh:
            json.dump(self._metrics, fh, indent=2, sort_keys=True)
        call_graph_serialized = {
            caller: sorted(callees) for caller, callees in self._call_graph.items()
        }
        _RELEVANCY_CALL_GRAPH_FILE.parent.mkdir(parents=True, exist_ok=True)
        with _RELEVANCY_CALL_GRAPH_FILE.open("w", encoding="utf-8") as fh:
            json.dump(call_graph_serialized, fh, indent=2, sort_keys=True)

    def _install_import_hook(self) -> None:
        if getattr(builtins, "_relevancy_radar_original_import", None):
            return

        original_import = builtins.__import__
        radar = self

        def tracked_import(name, globals=None, locals=None, fromlist=(), level=0):
            root = name.split(".")[0]
            info = radar._metrics.setdefault(
                root,
                {
                    "imports": 0.0,
                    "executions": 0.0,
                    "impact": 0.0,
                    "output_impact": 0.0,
                },
            )
            info["imports"] = float(info.get("imports", 0.0)) + 1.0
            return original_import(name, globals, locals, fromlist, level)

        builtins._relevancy_radar_original_import = original_import
        builtins.__import__ = tracked_import

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def track_usage(self, module_name: str, impact: float = 0.0) -> None:
        """Record an execution event for ``module_name`` and persist metrics."""

        info = self._metrics.setdefault(
            module_name,
            {"imports": 0.0, "executions": 0.0, "impact": 0.0, "output_impact": 0.0},
        )
        info["executions"] = float(info.get("executions", 0.0)) + 1.0
        info["impact"] = float(info.get("impact", 0.0)) + float(impact)
        try:
            caller_frame = inspect.stack()[1]
            caller = caller_frame.frame.f_globals.get("__name__")
        except Exception:
            caller = None
        finally:
            try:
                del caller_frame
            except Exception:
                pass
        if caller:
            self._call_graph.setdefault(caller, set()).add(module_name)
        self._persist_metrics()

    def record_output_impact(self, module_name: str, impact: float) -> None:
        """Record that ``module_name`` contributed ``impact`` to the final output."""

        info = self._metrics.setdefault(
            module_name,
            {"imports": 0.0, "executions": 0.0, "impact": 0.0, "output_impact": 0.0},
        )
        info["output_impact"] = float(info.get("output_impact", 0.0)) + float(impact)
        self._persist_metrics()

    def track(self, arg: Callable[..., Any] | str | None = None, *, module_name: str | None = None):
        """Return a decorator or context manager that records usage and impact."""

        radar = self

        class _Tracker(contextlib.ContextDecorator):
            def __init__(self, mod: str):
                self.mod = mod
                self.impact = 0.0

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                radar.track_usage(self.mod, float(self.impact))
                radar.record_output_impact(self.mod, float(self.impact))

            def record(self, impact: float) -> None:
                self.impact += float(impact)

        if callable(arg):
            func = arg
            name = module_name or f"{func.__module__}.{func.__qualname__}"
            sig = inspect.signature(func)

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                bound = sig.bind_partial(*args, **kwargs)
                impact = bound.arguments.get("impact")
                if impact is None:
                    impact = bound.arguments.get("roi_delta") or bound.arguments.get("roi")
                result = func(*args, **kwargs)
                res_impact = _extract_impact(result)
                final = impact if impact is not None else res_impact
                if final is None:
                    final = 0.0
                radar.track_usage(name, float(final))
                record_output_impact(name, float(final))
                return result

            return wrapper

        if isinstance(arg, str) and module_name is None:
            return _Tracker(arg)

        if arg is None:
            def decorator(func: Callable[..., Any]):
                return radar.track(func, module_name=module_name)

            if module_name is not None:
                return decorator
            return decorator

        raise TypeError("Unsupported arguments for track")

    def evaluate_relevance(
        self,
        compress_threshold: float,
        replace_threshold: float,
        impact_weight: float = 1.0,
        dep_graph: nx.DiGraph | None = None,
        core_modules: Iterable[str] | None = None,
    ) -> Dict[str, str]:
        """Return modules with scores below the provided thresholds.

        The score combines import counts, execution counts and the cumulative
        impact for each module. ``impact_weight`` scales the influence of the
        impact score relative to usage counts.

        Modules with a score of zero or less are tagged ``retire``. Modules with
        a score less than or equal to ``compress_threshold`` are tagged
        ``compress``.
        Modules with a score between ``compress_threshold`` and
        ``replace_threshold`` (inclusive) are tagged ``replace``.
        """

        results: Dict[str, str] = {}
        for mod, counts in self._metrics.items():
            impact_val = float(counts.get("impact", 0.0))
            score = (
                float(counts.get("imports", 0.0))
                + float(counts.get("executions", 0.0))
                + impact_weight * impact_val
            )
            if score <= 0:
                results[mod] = "retire"
            elif score <= compress_threshold:
                results[mod] = "compress"
            elif score <= replace_threshold:
                results[mod] = "replace"

        if dep_graph is not None:
            core_modules = list(core_modules or ["menace_master", "run_autonomous"])
            core_nodes = {m.replace(".", "/") for m in core_modules}
            reachable: set[str] = set()
            for core in core_nodes:
                if core in dep_graph:
                    reachable.add(core)
                    reachable.update(nx.descendants(dep_graph, core))
            for mod in list(results):
                node = mod.replace(".", "/")
                if node not in reachable:
                    results[mod] = "retire"

        self._persist_metrics()
        return results

    def evaluate_final_contribution(
        self,
        compress_threshold: float,
        replace_threshold: float,
        impact_weight: float = 1.0,
        core_modules: Iterable[str] | None = None,
    ) -> Dict[str, str]:
        """Return dependency-aware relevancy flags using the recorded call graph."""

        results: Dict[str, str] = {}
        for mod, counts in self._metrics.items():
            impact_val = float(counts.get("impact", 0.0)) + float(
                counts.get("output_impact", 0.0)
            )
            score = (
                float(counts.get("imports", 0.0))
                + float(counts.get("executions", 0.0))
                + impact_weight * impact_val
            )
            if score <= 0:
                results[mod] = "retire"
            elif score <= compress_threshold:
                results[mod] = "compress"
            elif score <= replace_threshold:
                results[mod] = "replace"

        graph = nx.DiGraph()
        for caller, callees in self._call_graph.items():
            for callee in callees:
                graph.add_edge(caller, callee)

        core_modules = list(core_modules or ["menace_master", "run_autonomous"])
        reachable: set[str] = set(core_modules)
        for core in core_modules:
            if core in graph:
                reachable.update(nx.descendants(graph, core))

        for mod in list(results):
            if mod not in reachable:
                results[mod] = "retire"

        return results

    @staticmethod
    def flag_unused_modules(
        module_map: Iterable[str], impact_stats: Dict[str, float] | None = None
    ) -> Dict[str, str]:
        """Evaluate module usage and persist flags and orphan list.

        Parameters
        ----------
        module_map:
            Iterable of module identifiers to evaluate.
        """

        modules = list(module_map)
        usage_stats = load_usage_stats()
        try:
            db = RelevancyMetricsDB(_RELEVANCY_METRICS_DB)
            db_impacts = db.get_roi_deltas(modules)
        except Exception:
            db_impacts = {}
        if impact_stats:
            for mod, val in impact_stats.items():
                db_impacts[mod] = db_impacts.get(mod, 0.0) + float(val)
        flags = evaluate_relevancy(
            dict.fromkeys(modules), usage_stats, db_impacts
        )
        update_relevancy_metrics(flags)

        orphan_file = _ORPHAN_MODULES_FILE
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


# ---------------------------------------------------------------------------
# Module-level convenience helpers
# ---------------------------------------------------------------------------
_DEFAULT_RADAR: RelevancyRadar | None = None


def _get_default_radar() -> RelevancyRadar:
    """Return a lazily instantiated :class:`RelevancyRadar` instance."""

    global _DEFAULT_RADAR
    if _DEFAULT_RADAR is None:
        _DEFAULT_RADAR = RelevancyRadar()
    return _DEFAULT_RADAR


radar = _get_default_radar()


def track(func: Callable[..., Any] | None = None, *, module_name: str | None = None):
    """Proxy to :meth:`RelevancyRadar.track` on the default radar."""

    return radar.track(func, module_name=module_name)


def track_usage(module_name: str, impact: float = 0.0) -> None:
    """Record an execution event for ``module_name`` using the default radar."""

    radar = _get_default_radar()
    radar.track_usage(module_name, impact)


def record_output_impact(module_name: str, impact: float) -> None:
    """Record that ``module_name`` contributed to the final output."""

    radar = _get_default_radar()
    radar.record_output_impact(module_name, impact)


def evaluate_relevance(
    compress_threshold: float,
    replace_threshold: float,
    impact_weight: float = 1.0,
) -> Dict[str, str]:
    """Evaluate relevancy of tracked modules using the default radar.

    Parameters
    ----------
    compress_threshold:
        Score at or below which modules are flagged ``compress``.
    replace_threshold:
        Score at or below which modules are flagged ``replace``.

    Returns
    -------
    Dict[str, str]
        Mapping of module names to relevancy status (``retire``/``compress``/``replace``).
    """

    radar = _get_default_radar()
    return radar.evaluate_relevance(
        compress_threshold, replace_threshold, impact_weight
    )


def evaluate_final_contribution(
    compress_threshold: float,
    replace_threshold: float,
    impact_weight: float = 1.0,
    core_modules: Iterable[str] | None = None,
) -> Dict[str, str]:
    """Evaluate relevancy with dependency tracing using the default radar."""

    radar = _get_default_radar()
    return radar.evaluate_final_contribution(
        compress_threshold,
        replace_threshold,
        impact_weight,
        core_modules=core_modules,
    )


def scan(
    db_path: str | Path = _RELEVANCY_METRICS_DB,
    min_calls: int = 0,
    compress_ratio: float = 0.01,
    replace_ratio: float = 0.05,
    impact_weight: float = 1.0,
) -> Dict[str, str]:
    """Analyse the relevancy metrics database and return flags for modules.

    The analysis uses simple heuristics:

    - ``retire``: modules invoked ``min_calls`` times or fewer.
    - ``compress``: modules whose combined call and impact ratios fall below
      ``compress_ratio``.
    - ``replace``: modules with combined ratios below ``replace_ratio``.

    Parameters
    ----------
    db_path:
        Path to the SQLite metrics database generated by
        :class:`~relevancy_metrics_db.RelevancyMetricsDB`.
    min_calls:
        Maximum number of invocations considered "unused" when evaluating
        retirement.
    compress_ratio:
        Threshold combined ratio (relative to the totals) at which modules are
        flagged for compression.
    replace_ratio:
        Threshold combined ratio at which modules are suggested for replacement.
    impact_weight:
        Multiplier applied to the impact ratio when computing the combined
        score.

    Returns
    -------
    Dict[str, str]
        Mapping of module names to ``retire``, ``compress`` or ``replace``
        flags. Modules not meeting any heuristic are omitted.
    """

    db_path = Path(db_path)
    if not db_path.exists():
        return {}

    LOCAL_TABLES.add("module_metrics")
    base_router = GLOBAL_ROUTER
    close_router = False
    if base_router is None:
        base_router = init_db_router(
            "relevancy_radar", local_db_path=str(db_path), shared_db_path=str(db_path)
        )
        close_router = True
    conn = base_router.get_connection("module_metrics")
    try:
        rows = conn.execute(
            "SELECT module_name, call_count, roi_delta FROM module_metrics"
        ).fetchall()
    finally:
        if close_router:
            base_router.close()

    if not rows:
        return {}

    total_calls = sum(int(r[1] or 0) for r in rows)
    total_roi = sum(max(float(r[2] or 0.0), 0.0) for r in rows)
    flags: Dict[str, str] = {}
    for name, calls, roi in rows:
        call_count = int(calls or 0)
        roi_value = float(roi or 0.0)
        call_ratio = call_count / total_calls if total_calls else 0.0
        roi_pos = max(roi_value, 0.0)
        roi_ratio = roi_pos / total_roi if total_roi else 0.0
        score = call_ratio + impact_weight * roi_ratio

        if call_count <= min_calls:
            flags[str(name)] = "retire"
        elif score <= compress_ratio:
            flags[str(name)] = "compress"
        elif score <= replace_ratio:
            flags[str(name)] = "replace"

    return flags
