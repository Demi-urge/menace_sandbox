import argparse
import json
import logging
import os
from typing import Dict, List

from menace.environment_generator import generate_presets
from sandbox_runner.cli import _run_sandbox
from sandbox_runner import _sandbox_main
from metrics_dashboard import MetricsDashboard
from pathlib import Path
from threading import Thread


def _capture_run(preset: Dict[str, str], args: argparse.Namespace):
    holder = {}

    def wrapper(p: Dict[str, str], a: argparse.Namespace):
        holder['tracker'] = _sandbox_main(p, a)

    _run_sandbox(args, sandbox_main=wrapper)
    return holder.get('tracker')


def _diminishing_modules(
    history: Dict[str, List[float]],
    flagged: set[str],
    threshold: float,
    consecutive: int = 3,
) -> List[str]:
    flags: List[str] = []
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run sandbox cycles until diminishing returns across presets",
    )
    parser.add_argument(
        "--preset-count",
        type=int,
        default=3,
        help="number of environment presets per iteration",
    )
    parser.add_argument("--sandbox-data-dir", help="override sandbox data directory")
    parser.add_argument("--max-iterations", type=int, help="maximum iterations")
    parser.add_argument(
        "--dashboard-port",
        type=int,
        help="start MetricsDashboard on this port",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.dashboard_port:
        history_file = Path(args.sandbox_data_dir or "sandbox_data") / "roi_history.json"
        dash = MetricsDashboard(str(history_file))
        Thread(target=dash.run, kwargs={"port": args.dashboard_port}, daemon=True).start()

    module_history: Dict[str, List[float]] = {}
    flagged: set[str] = set()
    last_tracker = None
    iteration = 0

    while args.max_iterations is None or iteration < args.max_iterations:
        iteration += 1
        presets = generate_presets(args.preset_count)
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
        if last_tracker:
            new_flags = _diminishing_modules(module_history, flagged, last_tracker.diminishing())
            flagged.update(new_flags)
        if module_history and set(module_history) <= flagged:
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


if __name__ == "__main__":
    main()
