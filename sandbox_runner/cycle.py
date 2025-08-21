from __future__ import annotations

import logging
from pathlib import Path

from logging_utils import get_logger, setup_logging, log_record
import os
import json
import subprocess
import shutil
import time
import asyncio
import sys
import inspect
import threading
import uuid
from typing import Any, Dict, TYPE_CHECKING
from types import SimpleNamespace
from sandbox_settings import SandboxSettings
from log_tags import FEEDBACK, IMPROVEMENT_PATH, INSIGHT, ERROR_FIX
from memory_logging import log_with_tags
from memory_aware_gpt_client import ask_with_memory
from vector_service import Retriever, FallbackResult
try:  # pragma: no cover - optional dependency
    from vector_service import ErrorResult  # type: ignore
except Exception:  # pragma: no cover - fallback when unavailable
    class ErrorResult(Exception):
        """Fallback ErrorResult when vector service lacks explicit class."""

        pass

try:  # pragma: no cover - optional dependency
    from vector_service import PatchLogger, VectorServiceError  # type: ignore
except Exception:  # pragma: no cover - fallback when unavailable
    PatchLogger = object  # type: ignore

    class VectorServiceError(Exception):
        """Fallback VectorServiceError when vector service is unavailable."""

        pass

if TYPE_CHECKING:  # pragma: no cover - import heavy types only for checking
    from sandbox_runner import SandboxContext
    from roi_tracker import ROITracker

try:
    from radon.metrics import mi_visit  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    mi_visit = None  # type: ignore

try:
    from pylint.lint import Run as PylintRun  # type: ignore
    from pylint.reporters.text import TextReporter  # type: ignore
    from io import StringIO
except Exception:  # pragma: no cover - optional dependency
    PylintRun = None  # type: ignore
    TextReporter = None  # type: ignore

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore

# ``VectorMetricsDB`` was previously used for logging patch outcomes but has
# since been superseded by :class:`vector_service.patch_logger.PatchLogger`.
# The legacy import and global instance are removed in favor of using
# ``PatchLogger`` directly where needed.

logger = get_logger(__name__)

from analytics import adaptive_roi_model
from adaptive_roi_predictor import load_training_data

ADAPTIVE_ROI_RETRAIN_INTERVAL = int(
    os.getenv("ADAPTIVE_ROI_RETRAIN_INTERVAL", "20")
)

from metrics_exporter import (
    orphan_modules_reintroduced_total,
    orphan_modules_tested_total,
    orphan_modules_failed_total,
    orphan_modules_redundant_total,
    orphan_modules_legacy_total,
    orphan_modules_reclassified_total,
)
from relevancy_radar import (
    RelevancyRadar,
    track_usage as _radar_track_usage,
    evaluate_final_contribution,
    record_output_impact,
    radar,
)

from .environment import (
    SANDBOX_ENV_PRESETS,
    auto_include_modules,
    record_error,
    run_scenarios,
    ERROR_CATEGORY_COUNTS,
)
from .resource_tuner import ResourceTuner
from .orphan_discovery import (
    discover_recursive_orphans,
    append_orphan_cache,
    append_orphan_classifications,
    prune_orphan_cache,
    load_orphan_cache,
    load_orphan_traces,
    append_orphan_traces,
)
import orphan_analyzer
import module_graph_analyzer

_ENABLE_RELEVANCY_RADAR = os.getenv("SANDBOX_ENABLE_RELEVANCY_RADAR") == "1"


def _async_track_usage(module: str, impact: float | None = None) -> None:
    """Record ``module`` usage asynchronously if radar is enabled.

    ``impact`` denotes the ROI delta attributable to ``module``.
    """

    if not _ENABLE_RELEVANCY_RADAR:
        return

    impact_val = 0.0 if impact is None else float(impact)

    def _track() -> None:
        try:
            _radar_track_usage(module, impact_val)
            record_output_impact(module, impact_val)
        except Exception:
            pass

    try:
        threading.Thread(target=_track, daemon=True).start()
    except Exception:
        pass


def map_module_identifier(
    name: str, repo: Path, impact: float | None = None
) -> str:
    """Return canonical module identifier for ``name`` relative to ``repo``.

    ``impact`` specifies the ROI delta attributable to the module which will be
    forwarded to the relevancy radar.
    """

    base = name.split(":", 1)[0]
    try:
        rel = Path(base).resolve().relative_to(repo)
    except Exception:
        rel = Path(base)
    module_id = rel.with_suffix("").as_posix()
    record_output_impact(module_id, 0.0 if impact is None else float(impact))
    _radar_track_usage(module_id, 0.0 if impact is None else float(impact))
    return module_id


def _choose_suggestion(ctx: Any, module: str) -> str:
    """Return an offline patch suggestion for *module*."""
    try:
        if ctx.suggestion_db:
            sugg = ctx.suggestion_db.best_match(module)
            if sugg:
                return sugg
    except Exception:
        logger.exception("suggestion db lookup failed for %s", module)
    return ctx.suggestion_cache.get(module, "refactor for clarity")


async def _collect_plugin_metrics_async(
    plugins: list,
    prev_roi: float,
    roi: float,
    resources: Dict[str, float] | None,
) -> Dict[str, float]:
    """Gather metrics from plugins asynchronously."""

    tasks = []
    for func in plugins:
        module = sys.modules.get(func.__module__)
        async_fn = getattr(module, "collect_metrics_async", None)
        if callable(async_fn):
            tasks.append(async_fn(prev_roi, roi, resources))
        else:
            tasks.append(asyncio.to_thread(func, prev_roi, roi, resources))

    merged: Dict[str, float] = {}
    for res in await asyncio.gather(*tasks, return_exceptions=True):
        if isinstance(res, Exception):
            record_error(res)
            continue
        if isinstance(res, dict):
            for k, v in res.items():
                try:
                    merged[k] = float(v)
                except Exception:
                    merged[k] = 0.0
    return merged


def run_workflow_scenarios(
    workflow_db: str | Path,
    data_dir: str | Path = "sandbox_data",
    tracker: "ROITracker" | None = None,
) -> Dict[str, Dict[str, float]]:
    """Execute :func:`run_scenarios` for every workflow in ``workflow_db``.

    Parameters
    ----------
    workflow_db:
        Path to the workflow database.
    data_dir:
        Directory where the aggregated ROI-delta report will be written.
    tracker:
        Optional :class:`ROITracker` reused across scenario runs.

    Returns
    -------
    Dict[str, Dict[str, float]]
        Mapping of workflow identifiers to scenario ROI deltas.
    """

    from task_handoff_bot import WorkflowDB

    wf_db = WorkflowDB(Path(workflow_db))
    workflows = wf_db.fetch(limit=1000)
    report: Dict[str, Dict[str, float]] = {}

    for wf in workflows:
        tracker, _, summary = run_scenarios(wf, tracker=tracker)
        deltas = {
            scen: info.get("roi_delta", 0.0)
            for scen, info in summary.get("scenarios", {}).items()
        }
        wf_id = str(getattr(wf, "wid", getattr(wf, "id", "")))
        report[wf_id] = deltas

    out_path = Path(data_dir) / "scenario_deltas.json"
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
    except Exception:  # pragma: no cover - best effort
        logger.exception("failed to write scenario deltas")

    return report


@radar.track
def include_orphan_modules(ctx: "SandboxContext") -> None:
    """Discover orphan modules and feed viable ones into the workflow.

    The helper consults :func:`discover_recursive_orphans` and the legacy
    ``scripts.discover_isolated_modules`` script to find Python modules that are
    not referenced anywhere in the repository. Newly discovered modules are
    recorded in ``ctx.orphan_traces`` and passed to
    :func:`environment.auto_include_modules` for optional integration. Only
    modules that successfully pass the integration checks are added to
    ``ctx.module_map`` and pruned from the orphan cache.
    """

    settings = getattr(ctx, "settings", SandboxSettings())
    if not getattr(settings, "auto_include_isolated", False):
        return

    try:
        trace = discover_recursive_orphans(str(ctx.repo))
        try:
            from scripts.discover_isolated_modules import discover_isolated_modules
        except Exception:  # pragma: no cover - optional helper
            def discover_isolated_modules(*_args: Any, **_kwargs: Any) -> list[str]:
                return []

        module_map = set(getattr(ctx, "module_map", set()))
        traces = getattr(ctx, "orphan_traces", {})
        history_updates: Dict[str, Dict[str, Any]] = {}
        try:
            cached = load_orphan_cache(ctx.repo)
            for k, info in cached.items():
                cur = traces.get(k)
                if cur:
                    cur.update(info)
                else:
                    traces[k] = dict(info)
        except Exception:
            pass
        try:
            hist = load_orphan_traces(ctx.repo)
            for k, info in hist.items():
                cur = traces.setdefault(k, {"parents": []})
                if "classification_history" not in cur and info.get("classification_history"):
                    cur["classification_history"] = list(info.get("classification_history", []))
                if "roi_history" not in cur and info.get("roi_history"):
                    cur["roi_history"] = list(info.get("roi_history", []))
        except Exception:
            pass
        discovered: list[str] = []

        for name, info in trace.items():
            rel = Path(*name.split("."))
            path = (ctx.repo / rel).with_suffix(".py")
            if not path.exists():
                path = ctx.repo / rel / "__init__.py"
            if not path.exists():
                continue
            rel_path = path.relative_to(ctx.repo).as_posix()
            if rel_path in module_map:
                continue
            entry = traces.get(rel_path)
            cls = info.get("classification")
            if entry:
                parents = set(entry.get("parents", []))
                parents.update(info.get("parents", []))
                entry["parents"] = sorted(parents)
                if cls:
                    if entry.get("classification") != cls:
                        entry["classification"] = cls
                        entry["redundant"] = cls in {"legacy", "redundant"}
                        hist = entry.setdefault("classification_history", [])
                        if not hist or hist[-1] != cls:
                            hist.append(cls)
                            history_updates.setdefault(rel_path, {}).setdefault(
                                "classification_history", []
                            ).append(cls)
                    else:
                        entry["classification"] = cls
                        entry["redundant"] = cls in {"legacy", "redundant"}
                elif "redundant" in info:
                    entry["redundant"] = bool(info["redundant"])
            else:
                cls_val = info.get("classification", "candidate")
                traces[rel_path] = {
                    "parents": info.get("parents", []),
                    "classification": cls_val,
                    "redundant": info.get("redundant", False),
                    "classification_history": [cls_val],
                    "roi_history": [],
                }
                history_updates.setdefault(rel_path, {}).setdefault(
                    "classification_history", []
                ).append(cls_val)
                discovered.append(rel_path)

        # ensure callers can inspect the updated traces even if no inclusion runs
        ctx.orphan_traces = traces

        for rel_path in discover_isolated_modules(
            str(ctx.repo), recursive=settings.recursive_isolated
        ):
            if rel_path in module_map:
                continue
            traces.setdefault(rel_path, {"parents": []})
            discovered.append(rel_path)

        if discovered:
            logger.info(
                "orphan discovery",
                extra=log_record(
                    discovered=sorted(discovered),
                    parents={m: traces.get(m, {}).get("parents", []) for m in discovered},
                ),
            )


        candidate_mods: list[str] = []
        for m, info in traces.items():
            if m in module_map:
                continue
            cls = info.get("classification")
            path = ctx.repo / m
            try:
                cur_mtime = path.stat().st_mtime
            except Exception:
                cur_mtime = None

            flagged = info.get("failed") or (
                (info.get("redundant") or cls in {"redundant", "legacy"})
                and not getattr(settings, "test_redundant_modules", False)
            )
            if flagged:
                prev_mtime = info.get("mtime")
                if prev_mtime is None:
                    continue
                if cur_mtime is not None and cur_mtime > prev_mtime:
                    info.pop("failed", None)
                    info["redundant"] = False
                    info["classification"] = "candidate"
                    if cur_mtime is not None:
                        info["mtime"] = cur_mtime
                    candidate_mods.append(m)
                continue

            if not getattr(settings, "test_redundant_modules", False):
                if cls in {"redundant", "legacy"} or info.get("redundant"):
                    if cur_mtime is not None:
                        info["mtime"] = cur_mtime
                    continue
                if cls == "candidate":
                    oa = sys.modules.get("orphan_analyzer", orphan_analyzer)
                    try:
                        if getattr(oa, "analyze_redundancy", lambda _p: False)(path):
                            info["classification"] = "redundant"
                            info["redundant"] = True
                            if cur_mtime is not None:
                                info["mtime"] = cur_mtime
                            continue
                    except Exception:
                        pass
            candidate_mods.append(m)

        pre_mods = set(module_map)
        tested: Dict[str, list[str]] = {"added": [], "failed": [], "redundant": []}
        roi_map: Dict[str, list[float]] = {}
        if candidate_mods:
            clusters: Dict[int, list[str]] = {0: list(candidate_mods)}
            try:
                id_map = {map_module_identifier(m, ctx.repo): m for m in candidate_mods}
                graph = module_graph_analyzer.build_import_graph(ctx.repo)
                sub_graph = graph.subgraph(id_map.keys()).copy()
                mapping = module_graph_analyzer.cluster_modules(sub_graph)
                clusters = {}
                for mod in candidate_mods:
                    cid = mapping.get(map_module_identifier(mod, ctx.repo), 0)
                    clusters.setdefault(cid, []).append(mod)
            except Exception:
                logger.exception("module clustering failed")
            for cid in sorted(clusters):
                mods = clusters[cid]
                logger.info(
                    "orphan cluster integration",
                    extra=log_record(cluster=cid, modules=mods),
                )
                try:
                    cluster_tracker, cluster_tested = auto_include_modules(
                        mods, recursive=True, validate=True
                    )
                except Exception as exc:
                    record_error(exc)
                    for m in mods:
                        _async_track_usage(m)
                    continue
                for k in tested:
                    tested[k].extend(cluster_tested.get(k, []))
                if cluster_tracker:
                    try:
                        ctx.tracker.merge_history(cluster_tracker)
                    except Exception:
                        logger.exception("failed to merge orphan metrics")
                    deltas = getattr(cluster_tracker, "module_deltas", {})
                    for m, vals in deltas.items():
                        roi_map.setdefault(m, []).extend(vals)
                    for m in mods:
                        impact_vals = deltas.get(m)
                        impact = (
                            float(sum(float(x) for x in impact_vals))
                            if impact_vals
                            else None
                        )
                        _async_track_usage(m, impact)
                else:
                    for m in mods:
                        _async_track_usage(m)

        added = set(tested.get("added", []))
        failed = set(tested.get("failed", []))
        redundant = set(tested.get("redundant", []))

        pre_mods.difference_update(failed | redundant)
        module_map = pre_mods | added
        ctx.module_map = module_map

        post_cache: Dict[str, Dict[str, Any]] = {}
        try:
            post_cache = load_orphan_cache(ctx.repo)
        except Exception:
            pass

        for m in candidate_mods:
            entry = traces.setdefault(m, {"parents": []})
            cache_info = post_cache.get(m, {})
            prev_cls = entry.get("classification")
            cls = cache_info.get("classification", prev_cls or "candidate")
            entry["classification"] = cls
            entry["redundant"] = cls != "candidate"
            if m in failed:
                entry["failed"] = True
            else:
                entry.pop("failed", None)
            try:
                mtime = (ctx.repo / m).stat().st_mtime
            except Exception:
                mtime = None
            if m in failed or m in redundant:
                if mtime is not None:
                    entry["mtime"] = mtime
            else:
                entry.pop("mtime", None)
            try:
                if prev_cls == "legacy" and cls != "legacy":
                    orphan_modules_legacy_total.dec(1)
                elif prev_cls != "legacy" and cls == "legacy":
                    orphan_modules_legacy_total.inc(1)
            except Exception:
                pass
            if prev_cls is not None and prev_cls != cls:
                try:
                    orphan_modules_reclassified_total.inc(1)
                except Exception:
                    pass

            hist = entry.setdefault("classification_history", [])
            if not hist or hist[-1] != cls:
                hist.append(cls)
                history_updates.setdefault(m, {}).setdefault(
                    "classification_history", []
                ).append(cls)
            roi_vals = roi_map.get(m)
            if roi_vals:
                entry.setdefault("roi_history", []).extend(
                    float(x) for x in roi_vals
                )
                history_updates.setdefault(m, {}).setdefault(
                    "roi_history", []
                ).extend(float(x) for x in roi_vals)
        
        try:
            cache_updates = {
                k: {
                    "parents": v.get("parents", []),
                    "classification": v.get("classification", "candidate"),
                    "redundant": v.get("redundant", False),
                    **({"failed": True} if v.get("failed") else {}),
                    **({"mtime": v.get("mtime")} if v.get("mtime") is not None else {}),
                }
                for k, v in traces.items()
            }
            append_orphan_cache(ctx.repo, cache_updates)
            try:
                explicit_path = ctx.repo / "sandbox_data" / "orphan_modules.json"
                existing = {}
                if explicit_path.exists():
                    existing = json.loads(explicit_path.read_text()) or {}
                existing.update(cache_updates)
                explicit_path.parent.mkdir(parents=True, exist_ok=True)
                explicit_path.write_text(json.dumps(existing, indent=2, sort_keys=True))
            except Exception:
                pass
            class_entries = {
                k: {
                    "parents": v.get("parents", []),
                    "classification": v.get("classification", "candidate"),
                    "redundant": v.get("redundant", False),
                }
                for k, v in traces.items()
            }
            append_orphan_classifications(ctx.repo, class_entries)
            append_orphan_traces(ctx.repo, history_updates)
        except Exception:
            logger.exception("failed to record orphan traces")
        
        if added:
            try:
                prune_orphan_cache(ctx.repo, added, traces)
                for m in added:
                    cls = traces.get(m, {}).get("classification")
                    if cls == "legacy":
                        try:
                            orphan_modules_legacy_total.dec(1)
                        except Exception:
                            pass
                    elif cls == "redundant":
                        try:
                            orphan_modules_redundant_total.dec(1)
                        except Exception:
                            pass
                    traces.pop(m, None)
            except Exception:
                logger.exception("failed to prune orphan cache")
        
        ctx.orphan_traces = traces
        
        tested_count = len(added | failed | redundant)
        if tested_count:
            legacy = {m for m in redundant if traces.get(m, {}).get("classification") == "legacy"}
            pure_redundant = redundant - legacy
            try:
                orphan_modules_tested_total.inc(tested_count)
                orphan_modules_reintroduced_total.inc(len(added))
                orphan_modules_failed_total.inc(len(failed))
                orphan_modules_redundant_total.inc(len(pure_redundant))
                orphan_modules_legacy_total.inc(len(legacy))
            except Exception:
                pass
            logger.info(
                "isolated module tests",
                extra=log_record(
                    added=sorted(added),
                    failed=sorted(failed),
                    redundant=sorted(pure_redundant),
                    legacy=sorted(legacy),
                ),
            )
    except Exception as exc:
        record_error(exc)


@radar.track
def _sandbox_cycle_runner(
    ctx: "SandboxContext",
    section: str | None,
    snippet: str | None,
    tracker: "ROITracker",
    scenario: str | None = None,
) -> None:
    """Run one self-improvement cycle within the sandbox.

    The function coordinates resource tuning, code improvement and testing
    for the provided ``ctx``.  It updates ``SANDBOX_ENV_PRESETS`` and the
    environment variable of the same name, invokes the orchestrator,
    self-improvement engine and tester, analyses the sandbox and records
    metrics on ``tracker``.  ``ctx`` is mutated in place and patches may be
    written to disk.  Any raised exceptions are logged and the cycle
    continues unless ``tracker`` indicates convergence.
    """

    global SANDBOX_ENV_PRESETS
    from sandbox_runner import (
        build_section_prompt,
        GPT_SECTION_PROMPT_MAX_LENGTH,
        GPT_KNOWLEDGE_SERVICE,
    )

    knowledge_service = GPT_KNOWLEDGE_SERVICE

    if getattr(ctx, "patch_logger", None) is None:
        try:
            ctx.patch_logger = PatchLogger(patch_db=getattr(ctx, "patch_db", None))
        except VectorServiceError:
            ctx.patch_logger = None

    env_val = os.getenv("SANDBOX_ENV_PRESETS")
    if env_val:
        try:
            data = json.loads(env_val)
            if isinstance(data, dict):
                data = [data]
            SANDBOX_ENV_PRESETS = [dict(p) for p in data]
        except Exception as exc:
            record_error(exc)

    tuner = ResourceTuner()

    low_roi_streak = 0
    resilience_history: list[float] = []
    prev_res_avg: float | None = None
    failure_start: float | None = None
    start_roi = ctx.prev_roi
    last_metrics: Dict[str, float] | None = None
    tracker.register_metrics(
        "orphan_modules_tested",
        "orphan_modules_passed",
        "orphan_modules_failed",
        "orphan_modules_redundant",
        "orphan_modules_legacy",
    )
    for idx in range(ctx.cycles):
        cycle_start = time.perf_counter()
        # ensure orphan modules are processed before each cycle begins
        include_orphan_modules(ctx)
        if section:
            mapped_sec = map_module_identifier(section, ctx.repo, 0.0)
            if mapped_sec in ctx.meta_log.flagged_sections:
                logger.debug("section %s already complete", section)
                break

        logger.debug(
            "resource tuning start",
            extra={"cycle": idx, "prev_roi": ctx.prev_roi},
        )
        try:
            SANDBOX_ENV_PRESETS = tuner.adjust(tracker, SANDBOX_ENV_PRESETS)
            os.environ["SANDBOX_ENV_PRESETS"] = json.dumps(SANDBOX_ENV_PRESETS)
        except Exception as exc:
            record_error(exc)
        logger.info(
            "resource tuning complete",
            extra=log_record(cycle=idx, presets=SANDBOX_ENV_PRESETS),
        )
        logger.info("sandbox cycle %d starting", idx)
        logger.info("orchestrator run", extra=log_record(cycle=idx))
        try:
            ctx.orchestrator.run_cycle(ctx.models)
        except Exception as exc:
            record_error(exc)
        logger.info("patch engine start", extra=log_record(cycle=idx))
        try:
            result = ctx.improver.run_cycle()
        except Exception as exc:
            record_error(exc)
            result = SimpleNamespace(roi=None)
        warnings = getattr(result, "warnings", None)
        if warnings:
            logger.warning("improvement warnings", extra=log_record(warnings=warnings))
        logger.info(
            "patch engine complete",
            extra=log_record(cycle=idx, roi=result.roi.roi if result.roi else 0.0),
        )
        logger.info("tester run", extra=log_record(cycle=idx))
        try:
            ctx.tester.run_once()
            results = getattr(ctx.tester, "results", {}) or {}
        except Exception as exc:
            record_error(exc)
            results = {}
        tested: list[str] = []
        passed: list[str] = []
        failed: list[str] = []
        redundant: list[str] = []
        legacy: list[str] = []
        tested_count = passed_count = failed_count = redundant_count = legacy_count = 0
        if results:
            passed = results.get("orphan_passed_modules", [])
            failed = results.get("orphan_failed_modules", [])
            redundant = results.get("orphan_redundant_modules", [])
            legacy = results.get("orphan_legacy_modules", [])
            tested = sorted(set(passed) | set(failed) | set(redundant) | set(legacy))
            tested_count = len(tested)
            passed_count = len(passed)
            failed_count = len(failed)
            redundant_count = len(redundant)
            legacy_count = len(legacy)
            logger.info(
                "self test results",
                extra=log_record(
                    tested=tested,
                    passed=passed,
                    failed=failed,
                    redundant=sorted(redundant),
                    legacy=sorted(legacy),
                ),
            )
            orphan_modules_tested_total.inc(tested_count)
            orphan_modules_reintroduced_total.inc(passed_count)
            orphan_modules_failed_total.inc(failed_count)
            orphan_modules_redundant_total.inc(redundant_count)
            orphan_modules_legacy_total.inc(legacy_count)
            traces = getattr(ctx, "orphan_traces", {})
            for mod in tested:
                if mod not in traces:
                    continue
                entry = traces.setdefault(mod, {"parents": []})
                entry["tested"] = True
                entry["failed"] = mod in failed
                if mod in redundant:
                    entry["redundant"] = True
                if mod in legacy:
                    entry["classification"] = "legacy"
                    entry["redundant"] = True
            ctx.orphan_traces = traces
        logger.debug("sandbox analysis start", extra={"cycle": idx})
        try:
            ctx.sandbox.analyse_and_fix(limit=getattr(ctx, "patch_retries", 1))
        except TypeError:
            try:
                ctx.sandbox.analyse_and_fix()
            except Exception as exc:
                record_error(exc)
        except Exception as exc:
            record_error(exc)
        logger.info("patch application", extra=log_record(cycle=idx))
        roi = result.roi.roi if result.roi else 0.0
        logger.info(
            "roi calculated",
            extra={"cycle": idx, "roi": roi, "prev_roi": ctx.prev_roi},
        )
        if ctx.predicted_roi is not None:
            logger.info(
                "roi actual",
                extra={"iteration": idx, "predicted": ctx.predicted_roi, "actual": roi},
            )
            wf_id = getattr(ctx, "workflow_id", "_global")
            tracker.record_prediction(ctx.predicted_roi, roi, workflow_id=wf_id)
            wf_mae = tracker.workflow_mae(wf_id)
            wf_var = tracker.workflow_variance(wf_id)
            conf_val = tracker.workflow_confidence(wf_id)
            try:
                raroi = roi - getattr(ctx, "prev_roi", 0.0)
            except Exception:
                raroi = roi
            settings = getattr(ctx, "settings", SandboxSettings())
            tracker.raroi_borderline_threshold = settings.borderline_raroi_threshold
            tracker.confidence_threshold = settings.borderline_confidence_threshold
            needs_review = conf_val < tracker.confidence_threshold
            low_raroi = raroi < tracker.raroi_borderline_threshold
            logger.info(
                "workflow prediction evaluation",
                extra=log_record(
                    workflow=wf_id,
                    predicted=ctx.predicted_roi,
                    actual=roi,
                    mae=wf_mae,
                    variance=wf_var,
                    confidence=conf_val,
                    human_review=needs_review if needs_review else None,
                    borderline=low_raroi if low_raroi else None,
                ),
            )
            if needs_review or low_raroi:
                try:
                    tracker.borderline_bucket.add_candidate(
                        wf_id, float(raroi), conf_val
                    )
                except Exception:  # pragma: no cover - best effort
                    logger.exception(
                        "failed to enqueue workflow for borderline review",
                        extra=log_record(workflow=wf_id),
                    )
                logger.info(
                    "workflow queued for borderline evaluation",
                    extra=log_record(
                        workflow=wf_id,
                        confidence=conf_val,
                        raroi=raroi,
                        threshold=tracker.raroi_borderline_threshold,
                    ),
                )
                if settings.micropilot_mode == "auto":
                    try:
                        tracker.borderline_bucket.process(
                            getattr(ctx, "micro_pilot_evaluator", None),
                            raroi_threshold=tracker.raroi_borderline_threshold,
                            confidence_threshold=tracker.confidence_threshold,
                        )
                    except Exception:  # pragma: no cover - best effort
                        logger.exception("failed to process borderline candidates")
        if ctx.predicted_lucrativity is not None:
            tracker.record_metric_prediction(
                "projected_lucrativity", ctx.predicted_lucrativity, roi
            )
        if ctx.va_client and getattr(ctx.va_client, "open_run_id", None):
            if roi >= ctx.prev_roi:
                try:
                    ctx.va_client.resolve_run_log("roi_increased")
                except Exception:
                    logger.exception("visual agent log resolution failed")
            else:
                try:
                    ctx.va_client.resolve_run_log("reverted")
                    ctx.va_client.revert()
                except Exception as exc:  # pragma: no cover - best effort
                    logger.exception("visual agent revert failed: %s", exc)
        elif roi < ctx.prev_roi and ctx.va_client:
            logger.info("ROI decreased but no unresolved /run logs; skipping revert")
        recovery_time = 0.0
        if roi < ctx.prev_roi:
            if failure_start is None:
                failure_start = time.perf_counter()
        elif failure_start is not None:
            recovery_time = time.perf_counter() - failure_start
            failure_start = None
        mods, ctx.meta_log.last_patch_id = ctx.changed_modules(
            ctx.meta_log.last_patch_id
        )
        if section:
            mods = [m for m in mods if m == section.split(":", 1)[0]]
        roi_delta = roi - ctx.prev_roi
        mapped_pairs = [
            (m, map_module_identifier(m, ctx.repo, roi_delta)) for m in mods
        ]
        mapped_pairs = [
            (orig, mapped)
            for orig, mapped in mapped_pairs
            if mapped not in ctx.meta_log.flagged_sections
        ]
        mods = [orig for orig, _ in mapped_pairs]
        mapped_mods = [mapped for _, mapped in mapped_pairs]
        mapped_section = (
            map_module_identifier(section, ctx.repo, roi_delta) if section else None
        )
        resources = None
        if ctx.res_db is not None:
            try:
                df = ctx.res_db.history()
                cols = [
                    c
                    for c in ("cpu", "memory", "disk", "time", "gpu")
                    if c in df.columns
                ]
                if not df.empty and cols:
                    row = df.iloc[-1]
                    resources = {c: float(row[c]) for c in cols}
            except Exception:
                resources = None
        gpu_usage = 0.0
        if psutil is not None:
            try:
                temps = getattr(psutil, "sensors_temperatures", lambda: {})()
                for name, entries in temps.items():
                    if "gpu" in name.lower() and entries:
                        cur = getattr(entries[0], "current", None)
                        if cur is not None:
                            gpu_usage = float(cur)
                            break
            except Exception:
                logger.exception("psutil GPU sensor query failed")
        if not gpu_usage:
            try:
                if shutil.which("nvidia-smi"):
                    out = subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-gpu=utilization.gpu",
                            "--format=csv,noheader,nounits",
                        ],
                        text=True,
                    )
                    vals = [float(x) for x in out.splitlines() if x.strip()]
                    if vals:
                        gpu_usage = sum(vals) / len(vals)
            except Exception:
                logger.exception("nvidia-smi GPU utilization query failed")
        if not gpu_usage:
            try:
                import docker  # type: ignore

                cid = os.getenv("HOSTNAME")
                if cid:
                    client = docker.from_env()
                    c = client.containers.get(cid)
                    stats = c.stats(stream=False)
                    gstats = stats.get("gpu_stats")
                    if isinstance(gstats, list) and gstats:
                        vals = [float(g.get("utilization_gpu", 0) or 0) for g in gstats]
                        if vals:
                            gpu_usage = sum(vals) / len(vals)
            except Exception:
                logger.exception("docker GPU stats collection failed")
        if resources is None:
            resources = {}
        resources.setdefault("gpu", gpu_usage)
        for m in mods:
            ctx.module_counts[m] = ctx.module_counts.get(m, 0) + 1
        total_mods = sum(ctx.module_counts.values()) or 1
        import math

        probs = [c / total_mods for c in ctx.module_counts.values()]
        shannon = -sum(p * math.log2(p) for p in probs if p > 0)

        bandit_penalty = 0.0
        try:
            for m in mods:
                path = ctx.repo / m.split(":", 1)[0]
                if not path.exists() or path.suffix != ".py":
                    continue
                res = subprocess.run(
                    ["bandit", "-f", "json", "-q", str(path)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if res.stdout:
                    data = json.loads(res.stdout)
                    for issue in data.get("results", []):
                        sev = str(issue.get("issue_severity", "")).lower()
                        if sev == "low":
                            bandit_penalty += 5.0
                        elif sev == "medium":
                            bandit_penalty += 10.0
                        else:
                            bandit_penalty += 15.0
        except Exception:
            logger.exception("bandit scan failed")
        if not bandit_penalty:
            bandit_penalty = len(mods) * 5.0
        security_score = max(0.0, 100.0 - bandit_penalty)
        safety_rating = (
            security_score if roi >= ctx.prev_roi else max(0.0, security_score - 5)
        )

        adaptability = float(len(mods))
        antifragility = max(0.0, roi - ctx.prev_roi)
        efficiency_metric = (
            100.0 - float(resources.get("cpu", 0.0)) if resources else 100.0
        )
        flexibility = float(len(mods)) / float(total_mods)

        cov_path = ctx.repo / ".coverage"
        if cov_path.exists():
            try:
                from coverage import CoverageData

                cov = CoverageData(basename=str(cov_path))
                cov.read()
                exec_lines = 0
                total_lines = 0
                for m in mods:
                    path = ctx.repo / m.split(":", 1)[0]
                    if not path.exists():
                        continue
                    lines = cov.lines(str(path)) or []
                    exec_lines += len(lines)
                    try:
                        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                            total_lines += sum(1 for _ in fh)
                    except Exception:
                        logger.exception("failed counting lines for %s", path)
                if total_lines:
                    coverage_percent = 100.0 * exec_lines / float(total_lines)
                    adaptability = coverage_percent
                    flexibility = coverage_percent / 100.0
            except Exception:
                logger.exception("coverage analysis failed")
        projected_lucrativity = 0.0
        if ctx.pre_roi_bot:
            try:
                projected_lucrativity = ctx.pre_roi_bot.predict_metric(
                    "lucrativity",
                    [roi, security_score, safety_rating, adaptability],
                )
            except Exception:
                logger.exception("lucrativity prediction failed")
        patch_complexity = 0.0
        if ctx.patch_db_path.exists():
            try:
                import sqlite3

                with sqlite3.connect(
                    ctx.patch_db_path, check_same_thread=False
                ) as conn:
                    rows = conn.execute(
                        "SELECT complexity_after FROM patch_history ORDER BY id DESC LIMIT 5"
                    ).fetchall()
                if rows:
                    patch_complexity = sum(float(r[0] or 0.0) for r in rows) / len(rows)
            except Exception:
                patch_complexity = 0.0
        energy_consumption = 100.0 - efficiency_metric
        resilience = 100.0 if roi >= ctx.prev_roi else 50.0
        logger.info(
            "resilience calculated",
            extra={"cycle": idx, "roi": roi, "resilience": resilience},
        )
        network_latency = float(resources.get("time", 0.0)) if resources else 0.0
        throughput = 100.0 / (network_latency + 1.0)
        risk_index = max(0.0, 100.0 - (security_score + safety_rating) / 2.0)
        maintainability = 0.0
        code_quality = 0.0
        try:
            targets = (
                [ctx.repo / section.split(":", 1)[0]]
                if section
                else [ctx.repo / m.split(":", 1)[0] for m in mods]
            )
            targets = [t for t in targets if t.exists()]
            if targets:
                mi_total = 0.0
                cq_total = 0.0
                for t in targets:
                    try:
                        src = t.read_text(encoding="utf-8")
                    except Exception:
                        continue
                    if mi_visit:
                        try:
                            mi_total += float(mi_visit(src, False))
                        except Exception:
                            logger.exception("radon mi_visit failed for %s", t)
                    if PylintRun and TextReporter:
                        try:
                            buf = StringIO()
                            res = PylintRun(
                                ["--score=y", "--exit-zero", str(t)],
                                reporter=TextReporter(buf),
                                exit=False,
                            )
                            cq_total += float(
                                getattr(res.linter.stats, "global_note", 0.0)
                            )
                        except Exception:
                            logger.exception("pylint run failed for %s", t)
                if mi_total:
                    maintainability = mi_total / len(targets)
                if cq_total:
                    code_quality = cq_total / len(targets)
        except Exception as exc:
            record_error(exc)
        metrics = {
            "security_score": security_score,
            "safety_rating": safety_rating,
            "adaptability": adaptability,
            "antifragility": antifragility,
            "shannon_entropy": shannon,
            "efficiency": efficiency_metric,
            "flexibility": flexibility,
            "gpu_usage": gpu_usage,
            "projected_lucrativity": projected_lucrativity,
            "profitability": roi,
            "patch_complexity": patch_complexity,
            "energy_consumption": energy_consumption,
            "resilience": resilience,
            "network_latency": network_latency,
            "throughput": throughput,
            "risk_index": risk_index,
            "maintainability": maintainability,
            "code_quality": code_quality,
            "recovery_time": recovery_time,
            "orphan_modules_tested": float(tested_count),
            "orphan_modules_passed": float(passed_count),
            "orphan_modules_failed": float(failed_count),
            "orphan_modules_redundant": float(redundant_count),
            "orphan_modules_legacy": float(legacy_count),
        }
        if ctx.plugins:
            try:
                extra = asyncio.run(
                    _collect_plugin_metrics_async(
                        ctx.plugins, ctx.prev_roi, roi, resources
                    )
                )
            except Exception as exc:
                record_error(exc)
                extra = None
            if extra:
                metrics.update(extra)
        if ctx.extra_metrics:
            metrics.update(ctx.extra_metrics)
        # include error category summaries
        for cat, count in ERROR_CATEGORY_COUNTS.items():
            metrics[f"error_category_{cat}"] = float(count)
        ERROR_CATEGORY_COUNTS.clear()
        try:
            detections = ctx.dd_bot.scan()
        except Exception as exc:
            record_error(exc)
            detections = []
        metrics["discrepancy_count"] = len(detections)
        scenario_metrics = (
            {f"{k}:{scenario}": v for k, v in metrics.items()} if scenario else {}
        )
        tracker.register_metrics(*metrics.keys(), *scenario_metrics.keys())
        if scenario:
            tracker.register_metrics("synergy_safety_rating")
            tracker.synergy_metrics_history.setdefault(
                "synergy_safety_rating", []
            ).append(safety_rating)
        ctx.data_bot.collect(
            "sandbox",
            revenue=roi,
            expense=0.0,
            security_score=security_score,
            safety_rating=safety_rating,
            adaptability=adaptability,
            antifragility=antifragility,
            shannon_entropy=shannon,
            efficiency=efficiency_metric,
            flexibility=flexibility,
            gpu_usage=gpu_usage,
            projected_lucrativity=projected_lucrativity,
            profitability=roi,
            patch_complexity=patch_complexity,
            energy_consumption=energy_consumption,
            resilience=resilience,
            network_latency=network_latency,
            throughput=throughput,
            risk_index=risk_index,
            maintainability=maintainability,
            code_quality=code_quality,
        )
        name_list = [mapped_section] if section else mapped_mods
        vertex, curve, should_stop, entropy_ceiling = tracker.update(
            ctx.prev_roi, roi, name_list, resources, {**metrics, **scenario_metrics}
        )
        if entropy_ceiling:
            ctx.meta_log.ceiling(tracker.tolerance)

        roi_delta = 0.0
        if len(tracker.roi_history) >= 2:
            roi_delta = tracker.roi_history[-1] - tracker.roi_history[-2]
        elif tracker.roi_history:
            roi_delta = tracker.roi_history[-1]

        raroi_delta = 0.0
        if len(tracker.raroi_history) >= 2:
            raroi_delta = tracker.raroi_history[-1] - tracker.raroi_history[-2]
        elif tracker.raroi_history:
            raroi_delta = tracker.raroi_history[-1]

        wf_id = getattr(ctx, "workflow_id", "_global")
        conf_val = tracker.workflow_confidence(wf_id)
        ctx.foresight_tracker.record_cycle_metrics(
            wf_id,
            {
                "roi_deltas": roi_delta,
                "raroi_deltas": raroi_delta,
                "confidence": conf_val,
                "resilience": resilience,
                "scenario_degradation": metrics.get("scenario_degradation", 0.0),
            },
        )
        ctx.workflow_stable = ctx.foresight_tracker.is_stable(wf_id)
        last_metrics = metrics
        feat_vec = [
            roi,
            security_score,
            safety_rating,
            adaptability,
            antifragility,
            shannon,
            efficiency_metric,
            flexibility,
        ]
        if ctx.pre_roi_bot and getattr(ctx.pre_roi_bot, "prediction_manager", None):
            try:
                tracker.predict_all_metrics(
                    ctx.pre_roi_bot.prediction_manager, feat_vec
                )
            except Exception:
                logger.exception("metric prediction failed")
        ctx.meta_log.log_cycle(
            idx,
            roi,
            name_list,
            "self_improvement",
            warnings=warnings,
            exec_time=time.perf_counter() - cycle_start,
        )
        forecast, interval = tracker.forecast()
        mae = tracker.rolling_mae()
        reliability = tracker.reliability()
        synergy_rel = tracker.reliability(metric="synergy_roi")
        synergy_mae = tracker.synergy_reliability()
        rel = synergy_rel if section is None else reliability
        ctx.roi_tolerance = max(
            ctx.base_roi_tolerance * (1.0 - rel), ctx.base_roi_tolerance * 0.1
        )
        ctx.predicted_roi = forecast
        ctx.predicted_lucrativity = projected_lucrativity
        logger.info(
            "roi forecast",
            extra={
                "forecast": forecast,
                "interval": interval,
                "iteration": idx,
                "mae": mae,
                "reliability": reliability,
                "synergy_reliability": synergy_rel,
                "synergy_mae": synergy_mae,
            },
        )
        thr = tracker.diminishing()
        e_thr = ctx.settings.entropy_plateau_threshold or thr
        e_consec = ctx.settings.entropy_plateau_consecutive or 3
        flagged = ctx.meta_log.diminishing(
            thr, consecutive=e_consec, entropy_threshold=e_thr
        )
        if ctx.gpt_client:
            brainstorm_summary = "; ".join(ctx.brainstorm_history[-3:])
            early_exit = False
            for mod in flagged:
                if section and mod != section:
                    continue
                try:
                    module_name = mod.split(":", 1)[0]
                    memory_key = mod
                    insight = ""
                    if knowledge_service:
                        try:
                            insight = knowledge_service.get_recent_insights(module_name)
                        except Exception:
                            insight = ""
                    text = snippet or ""
                    path = ctx.repo / module_name
                    if path.exists():
                        text = path.read_text(encoding="utf-8")[:500]
                    prompt = build_section_prompt(
                        mod,
                        tracker,
                        text,
                        prior=brainstorm_summary if brainstorm_summary else None,
                        max_prompt_length=GPT_SECTION_PROMPT_MAX_LENGTH,
                    )
                    if insight:
                        prompt = f"{insight}\n\n{prompt}"
                    gpt_mem = getattr(ctx.gpt_client, "gpt_memory", None)
                    builder = getattr(ctx, "context_builder", None)
                    if builder is not None:
                        cb_session = uuid.uuid4().hex
                        try:
                            mem_ctx = builder.build(memory_key, session_id=cb_session)
                            if isinstance(mem_ctx, (FallbackResult, ErrorResult)):
                                mem_ctx = ""
                        except Exception:
                            mem_ctx = ""
                        if mem_ctx:
                            prompt += "\n\n### Memory\n" + mem_ctx
                    lkm = getattr(__import__("sys").modules.get("sandbox_runner"), "LOCAL_KNOWLEDGE_MODULE", None)
                    history = ctx.conversations.get(memory_key, [])
                    history_text = "\n".join(
                        f"{m.get('role')}: {m.get('content')}" for m in history
                    )
                    prompt_text = (
                        f"{history_text}\nuser: {prompt}" if history_text else prompt
                    )
                    resp = ask_with_memory(
                        ctx.gpt_client,
                        f"sandbox_runner.cycle.{memory_key}",
                        prompt_text,
                        memory=gpt_mem,
                        tags=[FEEDBACK, IMPROVEMENT_PATH, ERROR_FIX, INSIGHT],
                    )
                    suggestion = (
                        resp.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                        .strip()
                    )
                    history = history + [{"role": "user", "content": prompt}]
                    if suggestion:
                        history.append({"role": "assistant", "content": suggestion})
                        if lkm:
                            try:
                                lkm.log(
                                    prompt,
                                    suggestion,
                                    tags=[f"sandbox_runner.cycle.{memory_key}", FEEDBACK, IMPROVEMENT_PATH, ERROR_FIX, INSIGHT],
                                )
                            except Exception:
                                logger.exception("local knowledge logging failed for %s", mod)
                    if len(history) > 6:
                        history = history[-6:]
                    ctx.conversations[memory_key] = history
                except Exception:
                    logger.exception("gpt suggestion failed for %s", mod)
                    continue
                if not suggestion:
                    continue
                session_id = ""
                vectors: list[tuple[str, str, float]] = []
                retrieval_metadata: Dict[str, Dict[str, Any]] = {}
                retriever: Retriever | None = getattr(ctx, "retriever", None)
                if retriever is not None:
                    session_id = uuid.uuid4().hex
                    try:
                        hits = retriever.search(mod, top_k=1, session_id=session_id)
                        if isinstance(hits, (FallbackResult, ErrorResult)):
                            if isinstance(hits, FallbackResult):
                                logger.debug(
                                    "retriever returned fallback for %s: %s",
                                    mod,
                                    getattr(hits, "reason", ""),
                                )
                            hits = []
                        for h in hits:
                            origin = h.get("origin_db", "")
                            vid = str(h.get("record_id", ""))
                            score = float(h.get("score") or 0.0)
                            vectors.append((origin, vid, score))
                            retrieval_metadata[f"{origin}:{vid}"] = {
                                "license": h.get("license"),
                                "license_fingerprint": h.get("license_fingerprint"),
                                "semantic_alerts": h.get("semantic_alerts"),
                                "alignment_severity": h.get("alignment_severity"),
                            }
                    except Exception:
                        logger.debug("retriever lookup failed", exc_info=True)
                context_meta: Dict[str, Any] | None = None
                if session_id:
                    context_meta = {
                        "retrieval_session_id": session_id,
                        "retrieval_vectors": vectors,
                    }
                try:
                    target_path = ctx.repo / module_name
                    patch_id, reverted, _ = ctx.engine.apply_patch(
                        target_path,
                        suggestion,
                        reason=suggestion,
                        trigger="sandbox_runner",
                        context_meta=context_meta,
                    )
                    patch_logger = ctx.patch_logger
                    if patch_logger and session_id and vectors:
                        ids = {f"{o}:{v}": s for o, v, s in vectors}
                        try:
                            patch_logger.track_contributors(
                                ids,
                                bool(patch_id) and not reverted,
                                patch_id=str(patch_id or ""),
                                session_id=session_id,
                                retrieval_metadata=retrieval_metadata,
                            )
                        except VectorServiceError:
                            logger.debug("patch logging failed", exc_info=True)
                    logger.info(
                        "patch applied", extra={"module": mod, "patch_id": patch_id}
                    )
                    if gpt_mem:
                        try:
                            result_text = "success" if patch_id else "failure"
                            log_with_tags(
                                gpt_mem,
                                f"sandbox_runner.cycle.{memory_key}.patch_id",
                                str(patch_id),
                                tags=[f"sandbox_runner.cycle.{memory_key}", FEEDBACK, IMPROVEMENT_PATH, ERROR_FIX, INSIGHT],
                            )
                            log_with_tags(
                                gpt_mem,
                                f"sandbox_runner.cycle.{memory_key}.result",
                                result_text,
                                tags=[f"sandbox_runner.cycle.{memory_key}", FEEDBACK, IMPROVEMENT_PATH, ERROR_FIX, INSIGHT],
                            )
                        except Exception:
                            logger.exception("memory logging failed for %s", mod)
                except PermissionError as exc:
                    logger.error("patch permission denied for %s: %s", mod, exc)
                    raise
                except Exception as exc:
                    logger.exception("patch application failed for %s", mod)
                    patch_id = None
                try:
                    res = ctx.improver.run_cycle()
                    new_roi = res.roi.roi if res.roi else roi
                    roi_delta = new_roi - roi
                    mapped = map_module_identifier(mod, ctx.repo, roi_delta)
                    _, _, _, entropy_ceiling = tracker.update(
                        roi, new_roi, [mapped], resources
                    )
                    if entropy_ceiling:
                        ctx.meta_log.ceiling(tracker.tolerance)
                    ctx.meta_log.log_cycle(
                        idx,
                        new_roi,
                        [mapped],
                        "gpt4",
                        exec_time=0.0,
                    )
                    if roi_delta <= tracker.diminishing() and patch_id:
                        logger.info(
                            "rolling back patch",
                            extra={"module": mod, "patch_id": patch_id},
                        )
                        ctx.engine.rollback_patch(str(patch_id))
                        early_exit = True
                        patch_logger = ctx.patch_logger
                        if patch_logger and session_id and vectors:
                            ids = {f"{o}:{v}": s for o, v, s in vectors}
                            try:
                                patch_logger.track_contributors(
                                    ids,
                                    False,
                                    patch_id=str(patch_id),
                                    session_id=session_id,
                                    contribution=roi_delta,
                                    retrieval_metadata=retrieval_metadata,
                                )
                            except VectorServiceError:
                                logger.debug("patch logging failed", exc_info=True)
                    else:
                        patch_logger = ctx.patch_logger
                        if patch_logger and session_id and vectors and patch_id:
                            ids = {f"{o}:{v}": s for o, v, s in vectors}
                            try:
                                patch_logger.track_contributors(
                                    ids,
                                    True,
                                    patch_id=str(patch_id),
                                    session_id=session_id,
                                    contribution=roi_delta,
                                    retrieval_metadata=retrieval_metadata,
                                )
                            except VectorServiceError:
                                logger.debug("patch logging failed", exc_info=True)
                        roi = new_roi
                except Exception:
                    logger.exception("patch from gpt failed for %s", mod)
                    patch_logger = ctx.patch_logger
                    if patch_logger and session_id and vectors:
                        ids = {f"{o}:{v}": s for o, v, s in vectors}
                        try:
                            patch_logger.track_contributors(
                                ids,
                                False,
                                patch_id=str(patch_id or ""),
                                session_id=session_id,
                                contribution=0.0,
                                retrieval_metadata=retrieval_metadata,
                            )
                        except VectorServiceError:
                            logger.debug("patch logging failed", exc_info=True)
                    early_exit = True
                    continue
                patch_logger = ctx.patch_logger
                if patch_id is None and patch_logger and session_id and vectors:
                    ids = {f"{o}:{v}": s for o, v, s in vectors}
                    try:
                        patch_logger.track_contributors(
                            ids,
                            False,
                            patch_id=str(patch_id or ""),
                            session_id=session_id,
                            contribution=0.0,
                            retrieval_metadata=retrieval_metadata,
                        )
                    except VectorServiceError:
                        logger.debug("patch logging failed", exc_info=True)
                if early_exit:
                    break
            if early_exit:
                break
            delta = abs(roi - ctx.prev_roi)
            if delta <= tracker.diminishing():
                low_roi_streak += 1
                if ctx.gpt_client:
                    try:
                        summary = "; ".join(
                            f"{k}:{metrics.get(k)}" for k in sorted(metrics)
                        )
                        prior = "; ".join(ctx.brainstorm_history[-3:])
                        insight = ""
                        if knowledge_service:
                            try:
                                insight = knowledge_service.get_recent_insights(
                                    "brainstorm"
                                )
                            except Exception:
                                insight = ""
                        prompt = build_section_prompt(
                            "overall",
                            tracker,
                            f"ROI stalled. Current metrics: {summary}",
                            prior=prior if prior else None,
                            max_prompt_length=GPT_SECTION_PROMPT_MAX_LENGTH,
                        )
                        if insight:
                            prompt = f"{insight}\n\n{prompt}"
                        gpt_mem = getattr(ctx.gpt_client, "gpt_memory", None)
                        builder = getattr(ctx, "context_builder", None)
                        if builder is not None:
                            cb_session = uuid.uuid4().hex
                            try:
                                mem_ctx = builder.build("brainstorm", session_id=cb_session)
                                if isinstance(mem_ctx, (FallbackResult, ErrorResult)):
                                    mem_ctx = ""
                            except Exception:
                                mem_ctx = ""
                            if mem_ctx:
                                prompt += "\n\n### Memory\n" + mem_ctx
                        hist = ctx.conversations.get("brainstorm", [])
                        lkm = getattr(__import__("sys").modules.get("sandbox_runner"), "LOCAL_KNOWLEDGE_MODULE", None)
                        history_text = "\n".join(
                            f"{m.get('role')}: {m.get('content')}" for m in hist
                        )
                        prompt_text = (
                            f"{history_text}\nuser: {prompt}" if history_text else prompt
                        )
                        resp = ask_with_memory(
                            ctx.gpt_client,
                            "sandbox_runner.cycle.brainstorm",
                            prompt_text,
                            memory=gpt_mem,
                            tags=[FEEDBACK, IMPROVEMENT_PATH, ERROR_FIX, INSIGHT],
                        )
                        idea = (
                            resp.get("choices", [{}])[0]
                            .get("message", {})
                            .get("content", "")
                            .strip()
                        )
                        hist = hist + [{"role": "user", "content": prompt}]
                        if idea:
                            ctx.brainstorm_history.append(idea)
                            hist.append({"role": "assistant", "content": idea})
                            logger.info("brainstorm", extra={"idea": idea})
                            if lkm:
                                try:
                                    lkm.log(
                                        prompt,
                                        idea,
                                        tags=[FEEDBACK, IMPROVEMENT_PATH, ERROR_FIX, INSIGHT],
                                    )
                                except Exception:
                                    logger.exception("local knowledge logging failed")
                        if len(hist) > 6:
                            hist = hist[-6:]
                        ctx.conversations["brainstorm"] = hist
                    except Exception:
                        logger.exception("brainstorming failed during stall")
                ctx.prev_roi = roi
                continue
            else:
                low_roi_streak = 0

            resilience_history.append(resilience)
            res_window = resilience_history[-3:]
            curr_avg = sum(res_window) / len(res_window)
            resilience_drop = False
            if (
                prev_res_avg is not None
                and prev_res_avg - curr_avg > tracker.diminishing()
            ):
                resilience_drop = True
            prev_res_avg = curr_avg

            brainstorm_now = False
            if ctx.brainstorm_interval and (idx + 1) % ctx.brainstorm_interval == 0:
                brainstorm_now = True
            elif low_roi_streak >= ctx.brainstorm_retries > 0:
                brainstorm_now = True
            elif resilience_drop:
                brainstorm_now = True
                try:
                    ctx.meta_log.log_cycle(
                        idx,
                        roi,
                        name_list,
                        "resilience_brainstorm",
                        exec_time=0.0,
                    )
                except Exception:
                    logger.exception("resilience brainstorm logging failed")
            if brainstorm_now:
                try:
                    summary = "; ".join(ctx.brainstorm_history[-3:])
                    insight = ""
                    if knowledge_service:
                        try:
                            insight = knowledge_service.get_recent_insights("brainstorm")
                        except Exception:
                            insight = ""
                    prompt = build_section_prompt(
                        "overall",
                        tracker,
                        "Brainstorm high level improvements to increase ROI.",
                        prior=summary if summary else None,
                        max_prompt_length=GPT_SECTION_PROMPT_MAX_LENGTH,
                    )
                    if insight:
                        prompt = f"{insight}\n\n{prompt}"
                    builder = getattr(ctx, "context_builder", None)
                    if builder is not None:
                        cb_session = uuid.uuid4().hex
                        try:
                            mem_ctx = builder.build("brainstorm", session_id=cb_session)
                            if isinstance(mem_ctx, (FallbackResult, ErrorResult)):
                                mem_ctx = ""
                        except Exception:
                            mem_ctx = ""
                        if mem_ctx:
                            prompt += "\n\n### Memory\n" + mem_ctx
                    hist = ctx.conversations.get("brainstorm", [])
                    lkm = getattr(__import__("sys").modules.get("sandbox_runner"), "LOCAL_KNOWLEDGE_MODULE", None)
                    history_text = "\n".join(
                        f"{m.get('role')}: {m.get('content')}" for m in hist
                    )
                    prompt_text = (
                        f"{history_text}\nuser: {prompt}" if history_text else prompt
                    )
                    resp = ask_with_memory(
                        ctx.gpt_client,
                        "sandbox_runner.cycle.brainstorm",
                        prompt_text,
                        memory=gpt_mem,
                        tags=[FEEDBACK, IMPROVEMENT_PATH, ERROR_FIX, INSIGHT],
                    )
                    idea = (
                        resp.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )
                    hist = hist + [{"role": "user", "content": prompt}]
                    if idea:
                        ctx.brainstorm_history.append(idea)
                        hist.append({"role": "assistant", "content": idea})
                        logger.info("brainstorm", extra={"idea": idea})
                        if lkm:
                            try:
                                lkm.log(
                                    prompt,
                                    idea,
                                    tags=[FEEDBACK, IMPROVEMENT_PATH, ERROR_FIX, INSIGHT],
                                )
                            except Exception:
                                logger.exception("local knowledge logging failed")
                    if len(hist) > 6:
                        hist = hist[-6:]
                    ctx.conversations["brainstorm"] = hist
                except Exception:
                    logger.exception("brainstorming failed")
                low_roi_streak = 0
        elif ctx.offline_suggestions:
            early_exit = False
            for mod in flagged:
                if section and mod != section:
                    continue
                module_name = mod.split(":", 1)[0]
                suggestion = _choose_suggestion(ctx, module_name)
                session_id = ""
                vectors: list[tuple[str, str, float]] = []
                retrieval_metadata: Dict[str, Dict[str, Any]] = {}
                retriever: Retriever | None = getattr(ctx, "retriever", None)
                if retriever is not None:
                    session_id = uuid.uuid4().hex
                    try:
                        hits = retriever.search(mod, top_k=1, session_id=session_id)
                        if isinstance(hits, (FallbackResult, ErrorResult)):
                            if isinstance(hits, FallbackResult):
                                logger.debug(
                                    "retriever returned fallback for %s: %s",
                                    mod,
                                    getattr(hits, "reason", ""),
                                )
                            hits = []
                        for h in hits:
                            origin = h.get("origin_db", "")
                            vid = str(h.get("record_id", ""))
                            score = float(h.get("score") or 0.0)
                            vectors.append((origin, vid, score))
                            retrieval_metadata[f"{origin}:{vid}"] = {
                                "license": h.get("license"),
                                "license_fingerprint": h.get("license_fingerprint"),
                                "semantic_alerts": h.get("semantic_alerts"),
                                "alignment_severity": h.get("alignment_severity"),
                            }
                    except Exception:
                        logger.debug("retriever lookup failed", exc_info=True)
                context_meta: Dict[str, Any] | None = None
                if session_id:
                    context_meta = {
                        "retrieval_session_id": session_id,
                        "retrieval_vectors": vectors,
                    }
                try:
                    target_path = ctx.repo / module_name
                    patch_id, reverted, _ = ctx.engine.apply_patch(
                        target_path,
                        suggestion,
                        reason=suggestion,
                        trigger="sandbox_runner",
                        context_meta=context_meta,
                    )
                    logger.info(
                        "patch applied", extra={"module": mod, "patch_id": patch_id}
                    )
                    patch_logger = ctx.patch_logger
                    if patch_logger and session_id and vectors:
                        ids = {f"{o}:{v}": s for o, v, s in vectors}
                        try:
                            patch_logger.track_contributors(
                                ids,
                                bool(patch_id) and not reverted,
                                patch_id=str(patch_id or ""),
                                session_id=session_id,
                                retrieval_metadata=retrieval_metadata,
                            )
                        except VectorServiceError:
                            logger.debug("patch logging failed", exc_info=True)
                except PermissionError as exc:
                    logger.error("patch permission denied for %s: %s", mod, exc)
                    raise
                except Exception as exc:
                    logger.exception("patch application failed for %s", mod)
                    patch_id = None
                try:
                    res = ctx.improver.run_cycle()
                    new_roi = res.roi.roi if res.roi else roi
                    roi_delta = new_roi - roi
                    mapped = map_module_identifier(mod, ctx.repo, roi_delta)
                    _, _, _, entropy_ceiling = tracker.update(
                        roi, new_roi, [mapped], resources
                    )
                    if entropy_ceiling:
                        ctx.meta_log.ceiling(tracker.tolerance)
                    ctx.meta_log.log_cycle(
                        idx,
                        new_roi,
                        [mapped],
                        "offline",
                        exec_time=0.0,
                    )
                    if roi_delta <= tracker.diminishing() and patch_id:
                        logger.info(
                            "rolling back patch",
                            extra={"module": mod, "patch_id": patch_id},
                        )
                        ctx.engine.rollback_patch(str(patch_id))
                        early_exit = True
                        patch_logger = ctx.patch_logger
                        if patch_logger and session_id and vectors:
                            ids = {f"{o}:{v}": s for o, v, s in vectors}
                            try:
                                patch_logger.track_contributors(
                                    ids,
                                    False,
                                    patch_id=str(patch_id),
                                    session_id=session_id,
                                    contribution=roi_delta,
                                    retrieval_metadata=retrieval_metadata,
                                )
                            except VectorServiceError:
                                logger.debug("patch logging failed", exc_info=True)
                    else:
                        patch_logger = ctx.patch_logger
                        if patch_logger and session_id and vectors and patch_id:
                            ids = {f"{o}:{v}": s for o, v, s in vectors}
                            try:
                                patch_logger.track_contributors(
                                    ids,
                                    True,
                                    patch_id=str(patch_id),
                                    session_id=session_id,
                                    contribution=roi_delta,
                                    retrieval_metadata=retrieval_metadata,
                                )
                            except VectorServiceError:
                                logger.debug("patch logging failed", exc_info=True)
                        roi = new_roi
                except Exception:
                    logger.exception("offline suggestion failed for %s", mod)
                    patch_logger = ctx.patch_logger
                    if patch_logger and session_id and vectors:
                        ids = {f"{o}:{v}": s for o, v, s in vectors}
                        try:
                            patch_logger.track_contributors(
                                ids,
                                False,
                                patch_id=str(patch_id or ""),
                                session_id=session_id,
                                contribution=0.0,
                                retrieval_metadata=retrieval_metadata,
                            )
                        except VectorServiceError:
                            logger.debug("patch logging failed", exc_info=True)
                    early_exit = True
                    continue
                patch_logger = ctx.patch_logger
                if patch_id is None and patch_logger and session_id and vectors:
                    ids = {f"{o}:{v}": s for o, v, s in vectors}
                    try:
                        patch_logger.track_contributors(
                            ids,
                            False,
                            patch_id=str(patch_id or ""),
                            session_id=session_id,
                            contribution=0.0,
                            retrieval_metadata=retrieval_metadata,
                        )
                    except VectorServiceError:
                        logger.debug("patch logging failed", exc_info=True)
                if early_exit:
                    break
            if early_exit:
                break
            delta = abs(roi - ctx.prev_roi)
            if delta <= tracker.diminishing():
                low_roi_streak += 1
            else:
                low_roi_streak = 0

            resilience_history.append(resilience)
            res_window = resilience_history[-3:]
            curr_avg = sum(res_window) / len(res_window)
            resilience_drop = False
            if (
                prev_res_avg is not None
                and prev_res_avg - curr_avg > tracker.diminishing()
            ):
                resilience_drop = True
            prev_res_avg = curr_avg
        if should_stop or abs(roi - ctx.prev_roi) < ctx.roi_tolerance:
            logger.info(
                "roi tracker stop",
                extra={"vertex": vertex, "curve": curve, "iteration": idx},
            )
            break
        if section and section in ctx.meta_log.flagged_sections:
            logger.debug("section %s finished after %d cycles", section, idx)
            break
        ctx.prev_roi = roi
        if roi > getattr(ctx, "best_roi", 0.0):
            ctx.best_roi = roi
            ctx.synergy_needed = False
        elif roi < getattr(ctx, "best_roi", 0.0) - tracker.diminishing():
            ctx.synergy_needed = True
        logger.info("cycle %d complete", idx)
        logger.info("cycle roi", extra={"iteration": idx, "roi": roi})

        try:
            RelevancyRadar.flag_unused_modules(getattr(ctx, "module_map", []))
        except Exception:
            logger.exception("relevancy radar flagging failed")

        if (idx + 1) % ADAPTIVE_ROI_RETRAIN_INTERVAL == 0:
            logger.info(
                "adaptive roi model retrain", extra=log_record(cycle=idx)
            )
            try:
                metrics = adaptive_roi_model.retrain()
                logger.info(
                    "adaptive roi model retrain complete",
                    extra=log_record(cycle=idx, **metrics),
                )
            except Exception:
                logger.exception("adaptive roi model retrain failed")

    # Persist merged ROI training data for the adaptive predictor
    try:
        load_training_data(
            tracker,
            evolution_path=ctx.repo / "evolution_history.db",
            evaluation_path=ctx.repo / "evaluation_history.db",
            roi_events_path=ctx.repo / "roi_events.db",
            output_path=ctx.repo / "sandbox_data/adaptive_roi.csv",
        )
    except Exception:
        logger.exception("adaptive roi data aggregation failed")

    flagged = []
    if ctx.adapt_presets:
        try:
            thr = tracker.diminishing()
            e_thr = ctx.settings.entropy_plateau_threshold or thr
            e_consec = ctx.settings.entropy_plateau_consecutive or 3
            flagged = ctx.meta_log.diminishing(
                thr, consecutive=e_consec, entropy_threshold=e_thr
            )
        except Exception:
            flagged = []
    if ctx.adapt_presets and flagged:
        try:
            logger.debug(
                "preset adaptation start",
                extra={"flagged": flagged},
            )
            from menace.environment_generator import adapt_presets

            SANDBOX_ENV_PRESETS = adapt_presets(tracker, SANDBOX_ENV_PRESETS)
            os.environ["SANDBOX_ENV_PRESETS"] = json.dumps(SANDBOX_ENV_PRESETS)
            logger.debug(
                "preset adaptation end",
                extra={"preset_count": len(SANDBOX_ENV_PRESETS)},
            )
        except Exception:
            logger.exception("preset adaptation failed")

    settings = getattr(ctx, "settings", SandboxSettings())
    if getattr(settings, "enable_relevancy_radar", False):
        try:
            flags = evaluate_final_contribution(
                settings.relevancy_radar_compress_ratio,
                settings.relevancy_radar_replace_ratio,
            )
            flag_path = ctx.repo / "sandbox_data" / "relevancy_flags.json"
            flag_path.parent.mkdir(parents=True, exist_ok=True)
            with flag_path.open("w", encoding="utf-8") as fh:
                json.dump(flags, fh, indent=2, sort_keys=True)
            logger.info("relevancy evaluation", extra=log_record(flags=flags))
        except Exception:
            logger.exception("relevancy radar final contribution evaluation failed")

    logger.info(
        "cycle summary",
        extra=log_record(
            start_roi=start_roi,
            final_roi=ctx.prev_roi,
            roi_delta=ctx.prev_roi - start_roi,
            metrics=last_metrics,
            flagged=list(ctx.meta_log.flagged_sections),
        ),
    )


if __name__ == "__main__":  # pragma: no cover - manual invocation
    setup_logging()
