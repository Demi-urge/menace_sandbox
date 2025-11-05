import argparse
import json
import logging
import os
from typing import Dict, List
import math

from menace.environment_generator import generate_presets
from sandbox_runner.cli import _run_sandbox
from sandbox_runner import _sandbox_main
from metrics_dashboard import MetricsDashboard
from dynamic_path_router import resolve_path
from pathlib import Path
from threading import Thread

def _capture_run(preset: Dict[str, str], args: argparse.Namespace):
    holder = {}

    def wrapper(p: Dict[str, str], a: argparse.Namespace, b: object) -> None:
        holder['tracker'] = _sandbox_main(p, a, b)

    _run_sandbox(args, sandbox_main=wrapper)
    return holder.get('tracker')


def _ema(values: list[float]) -> tuple[float, float]:
    alpha = 2.0 / (len(values) + 1)
    ema = values[0]
    ema_sq = values[0] ** 2
    for v in values[1:]:
        ema = alpha * v + (1 - alpha) * ema
        ema_sq = alpha * (v ** 2) + (1 - alpha) * ema_sq
    var = ema_sq - ema ** 2
    if var < 0:
        var = 0.0
    return ema, var ** 0.5


def _diminishing_modules(
    history: Dict[str, List[float]],
    flagged: set[str],
    threshold: float,
    consecutive: int = 3,
    std_threshold: float = 1e-3,
    *,
    confidence: float = 0.95,
) -> tuple[List[str], Dict[str, float]]:
    flags: List[str] = []
    confidences: Dict[str, float] = {}
    thr = float(threshold)
    for mod, vals in history.items():
        if mod in flagged or len(vals) < consecutive:
            continue
        window = vals[-consecutive:]
        ema, std = _ema(window)
        if std == 0:
            conf = 1.0 if abs(ema) <= thr else 0.0
        else:
            z = abs(ema) / (std / math.sqrt(len(window)))
            conf = math.erfc(z / math.sqrt(2))
        if abs(ema) <= thr and std <= std_threshold and conf >= confidence:
            flags.append(mod)
            confidences[mod] = conf
    return flags, confidences


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
        history_file = (
            Path(resolve_path(args.sandbox_data_dir or "sandbox_data"))
            / "roi_history.json"
        )
        dash = MetricsDashboard(str(history_file))
        Thread(
            target=dash.run, kwargs={"port": args.dashboard_port}, daemon=True
        ).start()

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
                sandbox_data_dir=resolve_path(args.sandbox_data_dir or "sandbox_data"),
                workflow_db="workflows.db",
                workflow_sim=False,
                preset_count=None,
                no_workflow_run=False,
                max_prompt_length=None,
                summary_depth=None,
                discover_orphans=getattr(args, "discover_orphans", False),
            )
            tracker = _capture_run(preset, run_args)
            if not tracker:
                continue
            last_tracker = tracker
            for mod, vals in tracker.module_deltas.items():
                module_history.setdefault(mod, []).extend(vals)
        if last_tracker:
            new_flags, _ = _diminishing_modules(
                module_history, flagged, last_tracker.diminishing()
            )
            flagged.update(new_flags)
        if module_history and set(module_history) <= flagged:
            break

    if last_tracker:
        print("=== Final Module Rankings ===")
        for mod, raroi, roi in last_tracker.rankings():
            print(f"{mod}: {raroi:.3f} (roi {roi:.3f})")
        print("=== Metrics ===")
        for name, vals in last_tracker.metrics_history.items():
            if vals:
                print(f"{name}: {vals[-1]:.3f}")
    else:
        print("No sandbox runs executed")


if __name__ == "__main__":
    main()
