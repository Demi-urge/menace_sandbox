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
from typing import Any, Dict, TYPE_CHECKING

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

logger = get_logger(__name__)

from .environment import SANDBOX_ENV_PRESETS, auto_include_modules
from .resource_tuner import ResourceTuner
from .orphan_discovery import discover_recursive_orphans


def map_module_identifier(name: str, repo: Path) -> str:
    """Return canonical module identifier for ``name`` relative to ``repo``."""
    base = name.split(":", 1)[0]
    try:
        rel = Path(base).resolve().relative_to(repo)
    except Exception:
        rel = Path(base)
    return rel.with_suffix("").as_posix()


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
            logger.exception("metrics plugin failed", exc_info=res)
            continue
        if isinstance(res, dict):
            for k, v in res.items():
                try:
                    merged[k] = float(v)
                except Exception:
                    merged[k] = 0.0
    return merged


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
    from sandbox_runner import build_section_prompt, GPT_SECTION_PROMPT_MAX_LENGTH

    env_val = os.getenv("SANDBOX_ENV_PRESETS")
    if env_val:
        try:
            data = json.loads(env_val)
            if isinstance(data, dict):
                data = [data]
            SANDBOX_ENV_PRESETS = [dict(p) for p in data]
        except Exception:
            logger.exception("failed to reload presets")

    tuner = ResourceTuner()

    low_roi_streak = 0
    resilience_history: list[float] = []
    prev_res_avg: float | None = None
    failure_start: float | None = None
    start_roi = ctx.prev_roi
    last_metrics: Dict[str, float] | None = None
    for idx in range(ctx.cycles):
        logger.debug(
            "resource tuning start",
            extra={"cycle": idx, "prev_roi": ctx.prev_roi},
        )
        try:
            SANDBOX_ENV_PRESETS = tuner.adjust(tracker, SANDBOX_ENV_PRESETS)
            os.environ["SANDBOX_ENV_PRESETS"] = json.dumps(SANDBOX_ENV_PRESETS)
        except Exception:
            logger.exception("resource tuning failed")
        logger.info(
            "resource tuning complete",
            extra=log_record(cycle=idx, presets=SANDBOX_ENV_PRESETS),
        )
        logger.info("sandbox cycle %d starting", idx)
        logger.info("orchestrator run", extra=log_record(cycle=idx))
        ctx.orchestrator.run_cycle(ctx.models)
        logger.info("patch engine start", extra=log_record(cycle=idx))
        result = ctx.improver.run_cycle()
        logger.info(
            "patch engine complete",
            extra=log_record(cycle=idx, roi=result.roi.roi if result.roi else 0.0),
        )
        logger.info("tester run", extra=log_record(cycle=idx))
        ctx.tester.run_once()
        logger.debug("sandbox analysis start", extra={"cycle": idx})
        try:
            ctx.sandbox.analyse_and_fix(limit=getattr(ctx, "patch_retries", 1))
        except TypeError:
            ctx.sandbox.analyse_and_fix()
        auto_iso = os.getenv("SANDBOX_AUTO_INCLUDE_ISOLATED", "").lower()
        discover_env = os.getenv("SANDBOX_DISCOVER_ISOLATED", "").lower()
        rec_env = os.getenv("SANDBOX_RECURSIVE_ISOLATED", "").lower()
        if auto_iso in ("1", "true", "yes"):
            try:
                from scripts.discover_isolated_modules import discover_isolated_modules

                recursive = rec_env not in ("0", "false", "no")
                paths = discover_isolated_modules(ctx.repo, recursive=recursive)

                module_map = set(getattr(ctx, "module_map", set()))
                traces = getattr(ctx, "orphan_traces", {})
                new_mods: list[str] = []
                for path in paths:
                    if path in module_map or path in traces:
                        continue
                    traces[path] = {"parents": [], "redundant": False}
                    new_mods.append(path)
                if new_mods:
                    try:
                        from self_test_service import SelfTestService

                        passed_mods: list[str] = []
                        failed_mods: list[str] = []

                        for mod in new_mods:
                            try:
                                svc = SelfTestService(
                                    pytest_args=mod,
                                    include_orphans=False,
                                    discover_orphans=False,
                                    discover_isolated=False,
                                    disable_auto_integration=True,
                                )
                                res = svc.run_once()
                                if res.get("failed"):
                                    failed_mods.append(mod)
                                else:
                                    passed_mods.append(mod)
                            except Exception:
                                logger.exception("self tests failed for %s", mod)
                                failed_mods.append(mod)

                        if passed_mods:
                            auto_include_modules(passed_mods, recursive=True)
                            module_map.update(passed_mods)

                        if failed_mods:
                            logger.warning(
                                "self tests failed for %s",
                                ", ".join(sorted(failed_mods)),
                            )
                            fail_cache = (
                                ctx.repo
                                / "sandbox_data"
                                / "failed_isolated_modules.json"
                            )
                            try:
                                existing = (
                                    json.loads(fail_cache.read_text())
                                    if fail_cache.exists()
                                    else []
                                )
                                existing = sorted(set(existing) | set(failed_mods))
                                fail_cache.parent.mkdir(parents=True, exist_ok=True)
                                fail_cache.write_text(json.dumps(existing, indent=2))
                            except Exception:
                                logger.exception(
                                    "failed to record self test failures",
                                )

                        if passed_mods and os.getenv(
                            "SANDBOX_CLEAN_ORPHANS", ""
                        ).lower() in ("1", "true", "yes"):
                            cache = ctx.repo / "sandbox_data" / "orphan_modules.json"
                            try:
                                data = (
                                    json.loads(cache.read_text())
                                    if cache.exists()
                                    else []
                                )
                                data = [p for p in data if p not in passed_mods]
                                cache.write_text(json.dumps(sorted(data), indent=2))
                            except Exception:
                                pass
                    except Exception:
                        logger.exception("isolated module self testing failed")
                ctx.module_map = module_map
                ctx.orphan_traces = traces
            except Exception:
                logger.exception("isolated module auto-inclusion failed")
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
            tracker.record_prediction(ctx.predicted_roi, roi)
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
        mapped_mods = [map_module_identifier(m, ctx.repo) for m in mods]
        mapped_section = map_module_identifier(section, ctx.repo) if section else None
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
                    import json

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
        except Exception:
            logger.exception("metric analysis failed")
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
        }
        if ctx.plugins:
            try:
                extra = asyncio.run(
                    _collect_plugin_metrics_async(
                        ctx.plugins, ctx.prev_roi, roi, resources
                    )
                )
            except Exception:
                logger.exception("metrics plugin collection failed")
                extra = None
            if extra:
                metrics.update(extra)
        if ctx.extra_metrics:
            metrics.update(ctx.extra_metrics)
        try:
            detections = ctx.dd_bot.scan()
        except Exception:
            logger.exception("discrepancy scan failed")
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
        vertex, curve, should_stop = tracker.update(
            ctx.prev_roi, roi, name_list, resources, {**metrics, **scenario_metrics}
        )
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
        ctx.meta_log.log_cycle(idx, roi, name_list, "self_improvement")
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
        flagged = ctx.meta_log.diminishing(tracker.diminishing())
        if ctx.gpt_client:
            brainstorm_summary = "; ".join(ctx.brainstorm_history[-3:])
            early_exit = False
            for mod in flagged:
                if section and mod != section:
                    continue
                try:
                    text = snippet or ""
                    path = ctx.repo / mod.split(":", 1)[0]
                    if path.exists():
                        text = path.read_text(encoding="utf-8")[:500]
                    prompt = build_section_prompt(
                        mod,
                        tracker,
                        text,
                        prior=brainstorm_summary if brainstorm_summary else None,
                        max_prompt_length=GPT_SECTION_PROMPT_MAX_LENGTH,
                    )
                    key = mod.split(":", 1)[0]
                    history = ctx.conversations.get(key, [])
                    resp = ctx.gpt_client.ask(
                        history + [{"role": "user", "content": prompt}]
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
                    if len(history) > 6:
                        history = history[-6:]
                    ctx.conversations[key] = history
                except Exception:
                    logger.exception("gpt suggestion failed for %s", mod)
                    continue
                if not suggestion:
                    continue
                try:
                    target_path = ctx.repo / module_name
                    patch_id, _reverted, _ = ctx.engine.apply_patch(
                        target_path, suggestion
                    )
                    logger.info(
                        "patch applied", extra={"module": mod, "patch_id": patch_id}
                    )
                except PermissionError as exc:
                    logger.error("patch permission denied for %s: %s", mod, exc)
                    raise
                except Exception as exc:
                    logger.exception("patch application failed for %s", mod)
                    patch_id = None
                try:
                    res = ctx.improver.run_cycle()
                    new_roi = res.roi.roi if res.roi else roi
                    mapped = map_module_identifier(mod, ctx.repo)
                    tracker.update(roi, new_roi, [mapped], resources)
                    ctx.meta_log.log_cycle(idx, new_roi, [mapped], "gpt4")
                    if new_roi - roi <= tracker.diminishing() and patch_id:
                        logger.info(
                            "rolling back patch",
                            extra={"module": mod, "patch_id": patch_id},
                        )
                        ctx.engine.rollback_patch(str(patch_id))
                        early_exit = True
                    else:
                        roi = new_roi
                except Exception:
                    logger.exception("patch from gpt failed for %s", mod)
                    early_exit = True
                    continue
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
                        prompt = build_section_prompt(
                            "overall",
                            tracker,
                            f"ROI stalled. Current metrics: {summary}",
                            prior=prior if prior else None,
                            max_prompt_length=GPT_SECTION_PROMPT_MAX_LENGTH,
                        )
                        hist = ctx.conversations.get("brainstorm", [])
                        resp = ctx.gpt_client.ask(
                            hist + [{"role": "user", "content": prompt}]
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
                    ctx.meta_log.log_cycle(idx, roi, name_list, "resilience_brainstorm")
                except Exception:
                    logger.exception("resilience brainstorm logging failed")
            if brainstorm_now:
                try:
                    summary = "; ".join(ctx.brainstorm_history[-3:])
                    prompt = build_section_prompt(
                        "overall",
                        tracker,
                        "Brainstorm high level improvements to increase ROI.",
                        prior=summary if summary else None,
                        max_prompt_length=GPT_SECTION_PROMPT_MAX_LENGTH,
                    )
                    hist = ctx.conversations.get("brainstorm", [])
                    resp = ctx.gpt_client.ask(
                        hist + [{"role": "user", "content": prompt}]
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
                try:
                    target_path = ctx.repo / module_name
                    patch_id, _reverted, _ = ctx.engine.apply_patch(
                        target_path, suggestion
                    )
                    logger.info(
                        "patch applied", extra={"module": mod, "patch_id": patch_id}
                    )
                except PermissionError as exc:
                    logger.error("patch permission denied for %s: %s", mod, exc)
                    raise
                except Exception as exc:
                    logger.exception("patch application failed for %s", mod)
                    patch_id = None
                try:
                    res = ctx.improver.run_cycle()
                    new_roi = res.roi.roi if res.roi else roi
                    mapped = map_module_identifier(mod, ctx.repo)
                    tracker.update(roi, new_roi, [mapped], resources)
                    ctx.meta_log.log_cycle(idx, new_roi, [mapped], "offline")
                    if new_roi - roi <= tracker.diminishing() and patch_id:
                        logger.info(
                            "rolling back patch",
                            extra={"module": mod, "patch_id": patch_id},
                        )
                        ctx.engine.rollback_patch(str(patch_id))
                        early_exit = True
                    else:
                        roi = new_roi
                except Exception:
                    logger.exception("offline suggestion failed for %s", mod)
                    early_exit = True
                    continue
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

    flagged = []
    if ctx.adapt_presets:
        try:
            flagged = ctx.meta_log.diminishing(tracker.diminishing())
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
