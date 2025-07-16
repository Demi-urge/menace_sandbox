from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, List
import tempfile
import shutil
from threading import Thread

from menace.metrics_dashboard import MetricsDashboard

from .environment import SANDBOX_ENV_PRESETS, simulate_full_environment

try:  # optional import for tests
    from menace.environment_generator import generate_presets
except Exception:  # pragma: no cover
    generate_presets = lambda n=None: [{}]  # type: ignore

logger = logging.getLogger(__name__)


def _run_sandbox(args: argparse.Namespace, sandbox_main=None) -> None:
    """Execute sandbox runs for one or multiple environment presets."""
    if sandbox_main is None:
        from sandbox_runner import _sandbox_main as sandbox_main

    presets = SANDBOX_ENV_PRESETS or [{}]
    if presets == [{}] and not os.environ.get("SANDBOX_ENV_PRESETS"):
        if os.getenv("SANDBOX_GENERATE_PRESETS", "1") != "0":
            try:
                from menace.environment_generator import generate_presets
                count = getattr(args, "preset_count", None)
                presets = generate_presets(count)
            except Exception:
                presets = [{}]
    if len(presets) > 1:
        from menace.roi_tracker import ROITracker

        summary = ROITracker()
        for idx, preset in enumerate(presets):
            tracker = simulate_full_environment(preset)
            delta = sum(tracker.roi_history)
            sec_hist = tracker.metrics_history.get("security_score", [])
            sec_val = sec_hist[-1] if sec_hist else 0.0
            summary.update(0.0, delta, modules=[f"preset_{idx}"], metrics={"security_score": sec_val})
        logger.info("sandbox presets complete", extra={"ranking": summary.rankings()})
        return

    preset = presets[0]
    sandbox_main(preset, args)


def rank_scenarios(paths: list[str]) -> None:
    """Print ROI/security rankings for multiple preset runs."""
    from menace.roi_tracker import ROITracker

    results: list[tuple[str, float, float]] = []
    for entry in paths:
        p = Path(entry)
        hist = p / "roi_history.json" if p.is_dir() else p
        name = p.name if p.is_dir() else p.stem
        tracker = ROITracker()
        try:
            tracker.load_history(str(hist))
        except Exception:
            logger.exception("failed to load history %s", hist)
            continue
        roi_total = sum(tracker.roi_history)
        sec_hist = tracker.metrics_history.get("security_score", [])
        sec_val = sec_hist[-1] if sec_hist else 0.0
        results.append((name, roi_total, sec_val))

    results.sort(key=lambda x: (x[1], x[2]), reverse=True)
    for name, roi_val, sec_val in results:
        print(f"{name} ROI={roi_val:.3f} security_score={sec_val:.3f}")


def rank_scenario_synergy(paths: list[str], metric: str = "roi") -> None:
    """Print synergy metric totals per scenario across runs."""

    from menace.roi_tracker import ROITracker
    from sandbox_runner.environment import aggregate_synergy_metrics

    metric_name = metric if str(metric).startswith("synergy_") else f"synergy_{metric}"

    tmp_dir = tempfile.mkdtemp(prefix="scen_syn_")
    files: list[str] = []
    try:
        for entry in paths:
            p = Path(entry)
            hist = p / "roi_history.json" if p.is_dir() else p
            name = p.name if p.is_dir() else p.stem
            tracker = ROITracker()
            try:
                tracker.load_history(str(hist))
            except Exception:
                logger.exception("failed to load history %s", hist)
                continue
            for scen, lst in tracker.scenario_synergy.items():
                vals = [float(d.get(metric_name, 0.0)) for d in lst]
                if not vals:
                    continue
                t = ROITracker()
                t.metrics_history[metric_name] = vals
                out = Path(tmp_dir) / f"{name}_{scen}.json"
                t.save_history(str(out))
                files.append(str(out))

        results = aggregate_synergy_metrics(files, metric)
        for name, val in results:
            print(f"{name} {metric}={val:.3f}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _capture_run(preset: dict[str, str], args: argparse.Namespace):
    """Run sandbox for ``preset`` and return the resulting tracker."""
    from sandbox_runner import _sandbox_main

    holder: dict[str, Any] = {}

    def wrapper(p: dict[str, str], a: argparse.Namespace):
        holder["tracker"] = _sandbox_main(p, a)

    _run_sandbox(args, sandbox_main=wrapper)
    return holder.get("tracker")


def _diminishing_modules(
    history: dict[str, list[float]],
    flagged: set[str],
    threshold: float,
    consecutive: int = 3,
) -> list[str]:
    """Return modules with ROI deltas below ``threshold``."""
    flags: list[str] = []
    thr = float(threshold)
    for mod, vals in history.items():
        if mod in flagged:
            continue
        if len(vals) >= consecutive and all(abs(v) <= thr for v in vals[-consecutive:]):
            flags.append(mod)
            continue
        if len(vals) >= 3 and (vals[-1] <= thr or abs(vals[-1]) < abs(vals[0]) * 0.1):
            flags.append(mod)
    return flags


def full_autonomous_run(args: argparse.Namespace) -> None:
    """Execute sandbox cycles until all modules show diminishing returns."""
    if getattr(args, "dashboard_port", None):
        history_file = Path(args.sandbox_data_dir or "sandbox_data") / "roi_history.json"
        dash = MetricsDashboard(str(history_file))
        Thread(target=dash.run, kwargs={"port": args.dashboard_port}, daemon=True).start()

    module_history: dict[str, list[float]] = {}
    flagged: set[str] = set()
    synergy_history: list[dict[str, float]] = []
    synergy_streak = 0
    last_tracker = None
    iteration = 0
    roi_cycles = getattr(args, "roi_cycles", 3)
    synergy_cycles = getattr(args, "synergy_cycles", 3)

    while args.max_iterations is None or iteration < args.max_iterations:
        iteration += 1
        presets = generate_presets(args.preset_count or 3)
        for preset in presets:
            os.environ["SANDBOX_ENV_PRESETS"] = json.dumps([preset])
            run_args = argparse.Namespace(
                sandbox_data_dir=args.sandbox_data_dir,
                workflow_db="workflows.db",
                workflow_sim=False,
                preset_count=None,
                no_workflow_run=False,
                max_prompt_length=None,
                summary_depth=None,
            )
            tracker = _capture_run(preset, run_args)
            if not tracker:
                continue
            last_tracker = tracker
            for mod, vals in tracker.module_deltas.items():
                module_history.setdefault(mod, []).extend(vals)
            syn_vals = {
                k: v[-1]
                for k, v in tracker.metrics_history.items()
                if k.startswith("synergy_") and v
            }
            if syn_vals:
                synergy_history.append(syn_vals)
                max_syn = max(abs(v) for v in syn_vals.values())
                if max_syn <= tracker.diminishing():
                    synergy_streak += 1
                else:
                    synergy_streak = 0
        if last_tracker:
            new_flags = _diminishing_modules(
                module_history, flagged, last_tracker.diminishing(), consecutive=roi_cycles
            )
            flagged.update(new_flags)
        if module_history and set(module_history) <= flagged and synergy_streak >= synergy_cycles:
            logger.info(
                "synergy convergence reached",
                extra={"iteration": iteration, "streak": synergy_streak},
            )
            break

    if last_tracker:
        print("=== Final Module Rankings ===")
        for mod, total in last_tracker.rankings():
            print(f"{mod}: {total:.3f}")
        print("=== Metrics ===")
        for name, vals in last_tracker.metrics_history.items():
            if vals:
                print(f"{name}: {vals[-1]:.3f}")
    else:
        print("No sandbox runs executed")


def run_complete(args: argparse.Namespace) -> None:
    """Run ``full_autonomous_run`` with explicitly supplied presets."""
    def _load(val: str) -> dict[str, Any]:
        if os.path.exists(val):
            with open(val, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return json.loads(val)

    presets = [_load(p) for p in args.presets]

    global generate_presets
    original = generate_presets
    generate_presets = lambda n=None: presets  # type: ignore
    try:
        full_autonomous_run(args)
    finally:
        generate_presets = original


def main(argv: List[str] | None = None) -> None:
    """Entry point for command line execution."""
    parser = argparse.ArgumentParser(description="Run Menace sandbox")
    parser.add_argument("--workflow-sim", action="store_true", help="simulate workflows instead of repo sections")
    parser.add_argument("--workflow-db", default="workflows.db", help="path to workflow database")
    parser.add_argument("--sandbox-data-dir", help="override data directory for sandbox mode")
    parser.add_argument("--preset-count", type=int, help="number of presets to generate when none are provided")
    parser.add_argument("--max-prompt-length", type=int, help="maximum characters for GPT prompts")
    parser.add_argument("--summary-depth", type=int, help="lines to keep when summarising snippets")
    parser.add_argument(
        "--offline-suggestions",
        action="store_true",
        help="use heuristic suggestions when GPT is unavailable",
    )
    parser.add_argument(
        "--suggestion-cache",
        help="path to JSON cache with offline suggestions",
    )
    parser.add_argument(
        "--no-workflow-run",
        action="store_true",
        help="skip workflow simulations after section cycles",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("SANDBOX_LOG_LEVEL", os.getenv("LOG_LEVEL", "INFO")),
        help="logging level for console output",
    )

    sub = parser.add_subparsers(dest="cmd")
    p_rank = sub.add_parser("rank-scenarios", help="rank preset runs")
    p_rank.add_argument("paths", nargs="+", help="run directories or history files")

    p_syn = sub.add_parser("rank-synergy", help="rank preset synergy metrics")
    p_syn.add_argument("paths", nargs="+", help="run directories or history files")
    p_syn.add_argument("--metric", default="roi", help="synergy metric name")

    p_scen = sub.add_parser(
        "rank-scenario-synergy", help="rank scenario synergy metrics"
    )
    p_scen.add_argument("paths", nargs="+", help="run directories or history files")
    p_scen.add_argument("--metric", default="roi", help="synergy metric name")

    p_autorun = sub.add_parser(
        "full-autonomous-run",
        help="iterate presets until ROI improvements fade",
    )
    p_autorun.add_argument("--max-iterations", type=int, help="maximum iterations")
    p_autorun.add_argument(
        "--dashboard-port",
        type=int,
        help="start MetricsDashboard on this port",
    )
    p_autorun.add_argument(
        "--roi-cycles",
        type=int,
        default=3,
        help="cycles below threshold before module convergence",
    )
    p_autorun.add_argument(
        "--synergy-cycles",
        type=int,
        default=3,
        help="cycles below threshold before synergy convergence",
    )

    p_complete = sub.add_parser(
        "run-complete",
        help="run full-autonomous loop with provided presets",
    )
    p_complete.add_argument(
        "presets",
        nargs="+",
        help="JSON strings or files defining environment presets",
    )
    p_complete.add_argument("--max-iterations", type=int, help="maximum iterations")
    p_complete.add_argument(
        "--dashboard-port",
        type=int,
        help="start MetricsDashboard on this port",
    )
    p_complete.add_argument(
        "--roi-cycles",
        type=int,
        default=3,
        help="cycles below threshold before module convergence",
    )
    p_complete.add_argument(
        "--synergy-cycles",
        type=int,
        default=3,
        help="cycles below threshold before synergy convergence",
    )

    args = parser.parse_args(argv)

    level_str = str(getattr(args, "log_level", "INFO"))
    try:
        level = int(level_str)
    except ValueError:
        level = getattr(logging, level_str.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.max_prompt_length is not None:
        os.environ["GPT_SECTION_PROMPT_MAX_LENGTH"] = str(args.max_prompt_length)
    if args.summary_depth is not None:
        os.environ["GPT_SECTION_SUMMARY_DEPTH"] = str(args.summary_depth)

    if getattr(args, "cmd", None) == "rank-scenarios":
        rank_scenarios(args.paths)
        return

    if getattr(args, "cmd", None) == "rank-synergy":
        from sandbox_runner.environment import aggregate_synergy_metrics

        results = aggregate_synergy_metrics(args.paths, args.metric)
        for name, val in results:
            print(f"{name} {args.metric}={val:.3f}")
        return

    if getattr(args, "cmd", None) == "rank-scenario-synergy":
        rank_scenario_synergy(args.paths, args.metric)
        return

    if getattr(args, "cmd", None) == "full-autonomous-run":
        full_autonomous_run(args)
        return

    if getattr(args, "cmd", None) == "run-complete":
        run_complete(args)
        return

    if args.workflow_sim:
        from sandbox_runner.environment import run_workflow_simulations

        run_workflow_simulations(args.workflow_db)
    else:
        _run_sandbox(args)

