from __future__ import annotations

import logging
import os
import subprocess
import shutil
import time
from typing import Any

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

logger = logging.getLogger(__name__)

from .environment import SANDBOX_ENV_PRESETS


def _sandbox_cycle_runner(
    ctx: Any,
    section: str | None,
    snippet: str | None,
    tracker: Any,
    scenario: str | None = None,
) -> None:
    """Execute a single improvement/test cycle."""

    from sandbox_runner import build_section_prompt, GPT_SECTION_PROMPT_MAX_LENGTH
    from sandbox_runner.metrics_plugins import collect_plugin_metrics

    low_roi_streak = 0
    resilience_history: list[float] = []
    prev_res_avg: float | None = None
    failure_start: float | None = None
    for idx in range(ctx.cycles):
        logger.info("sandbox cycle %d starting", idx)
        ctx.orchestrator.run_cycle(ctx.models)
        result = ctx.improver.run_cycle()
        ctx.tester._run_once()
        ctx.sandbox.analyse_and_fix(limit=getattr(ctx, "patch_retries", 1))
        roi = result.roi.roi if result.roi else 0.0
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
        mods, ctx.meta_log.last_patch_id = ctx.changed_modules(ctx.meta_log.last_patch_id)
        if section:
            mods = [m for m in mods if m == section.split(":", 1)[0]]
        resources = None
        if ctx.res_db is not None:
            try:
                df = ctx.res_db.history()
                cols = [c for c in ("cpu", "memory", "disk", "time", "gpu") if c in df.columns]
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
                pass
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
                pass
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
                pass
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
        safety_rating = security_score if roi >= ctx.prev_roi else max(0.0, security_score - 5)

        adaptability = float(len(mods))
        antifragility = max(0.0, roi - ctx.prev_roi)
        efficiency_metric = 100.0 - float(resources.get("cpu", 0.0)) if resources else 100.0
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
                        pass
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

                with sqlite3.connect(ctx.patch_db_path, check_same_thread=False) as conn:
                    rows = conn.execute(
                        "SELECT complexity_after FROM patch_history ORDER BY id DESC LIMIT 5"
                    ).fetchall()
                if rows:
                    patch_complexity = sum(float(r[0] or 0.0) for r in rows) / len(rows)
            except Exception:
                patch_complexity = 0.0
        energy_consumption = 100.0 - efficiency_metric
        resilience = 100.0 if roi >= ctx.prev_roi else 50.0
        network_latency = float(resources.get("time", 0.0)) if resources else 0.0
        throughput = 100.0 / (network_latency + 1.0)
        risk_index = max(0.0, 100.0 - (security_score + safety_rating) / 2.0)
        maintainability = 0.0
        code_quality = 0.0
        try:
            targets = (
                [ctx.repo / section.split(":", 1)[0]] if section else [ctx.repo / m.split(":", 1)[0] for m in mods]
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
                            pass
                    if PylintRun and TextReporter:
                        try:
                            buf = StringIO()
                            res = PylintRun(
                                ["--score=y", "--exit-zero", str(t)],
                                reporter=TextReporter(buf),
                                exit=False,
                            )
                            cq_total += float(getattr(res.linter.stats, "global_note", 0.0))
                        except Exception:
                            pass
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
            extra = collect_plugin_metrics(ctx.plugins, ctx.prev_roi, roi, resources)
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
        scenario_metrics = {f"{k}:{scenario}": v for k, v in metrics.items()} if scenario else {}
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
        name_list = [section] if section else mods
        vertex, curve, should_stop = tracker.update(
            ctx.prev_roi, roi, name_list, resources, {**metrics, **scenario_metrics}
        )
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
                tracker.predict_all_metrics(ctx.pre_roi_bot.prediction_manager, feat_vec)
            except Exception:
                logger.exception("metric prediction failed")
        ctx.meta_log.log_cycle(idx, roi, name_list, "self_improvement")
        forecast, interval = tracker.forecast()
        mae = tracker.rolling_mae()
        reliability = tracker.reliability()
        synergy_rel = tracker.reliability(metric="synergy_roi")
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
                    resp = ctx.gpt_client.ask(history + [{"role": "user", "content": prompt}])
                    suggestion = resp.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
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
                    target_path = ctx.repo / mod.split(":", 1)[0]
                    patch_id, _reverted, _ = ctx.engine.apply_patch(target_path, suggestion)
                    logger.info("patch applied", extra={"module": mod, "patch_id": patch_id})
                except Exception:
                    patch_id = None
                try:
                    res = ctx.improver.run_cycle()
                    new_roi = res.roi.roi if res.roi else roi
                    tracker.update(roi, new_roi, [mod], resources)
                    ctx.meta_log.log_cycle(idx, new_roi, [mod], "gpt4")
                    if new_roi - roi <= tracker.diminishing() and patch_id:
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
            if prev_res_avg is not None and prev_res_avg - curr_avg > tracker.diminishing():
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
                    resp = ctx.gpt_client.ask(hist + [{"role": "user", "content": prompt}])
                    idea = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
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
                suggestion = ctx.suggestion_cache.get(mod.split(":", 1)[0], "refactor for clarity")
                try:
                    target_path = ctx.repo / mod.split(":", 1)[0]
                    patch_id, _reverted, _ = ctx.engine.apply_patch(target_path, suggestion)
                    logger.info("patch applied", extra={"module": mod, "patch_id": patch_id})
                except Exception:
                    patch_id = None
                try:
                    res = ctx.improver.run_cycle()
                    new_roi = res.roi.roi if res.roi else roi
                    tracker.update(roi, new_roi, [mod], resources)
                    ctx.meta_log.log_cycle(idx, new_roi, [mod], "offline")
                    if new_roi - roi <= tracker.diminishing() and patch_id:
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
            if prev_res_avg is not None and prev_res_avg - curr_avg > tracker.diminishing():
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

    if section:
        try:
            from menace.environment_generator import adapt_presets

            global SANDBOX_ENV_PRESETS
            SANDBOX_ENV_PRESETS = adapt_presets(tracker, SANDBOX_ENV_PRESETS)
        except Exception:
            logger.exception("preset adaptation failed")

