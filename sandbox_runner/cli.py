from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, List
import tempfile
import shutil
import math
from scipy.stats import pearsonr, t
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


def _ema(values: list[float]) -> tuple[float, float]:
    """Return exponential moving average and std deviation for ``values``."""
    alpha = 2.0 / (len(values) + 1)
    ema = values[0]
    ema_sq = values[0] ** 2
    for v in values[1:]:
        ema = alpha * v + (1 - alpha) * ema
        ema_sq = alpha * (v ** 2) + (1 - alpha) * ema_sq
    var = ema_sq - ema ** 2
    if var < 1e-12:
        var = 0.0
    return ema, var ** 0.5


def _adaptive_threshold(values: list[float], window: int, factor: float = 2.0) -> float:
    """Return adaptive threshold based on recent variance.

    ``window`` specifies how many recent values to consider. ``factor`` scales
    the exponentially weighted standard deviation.
    """
    if not values:
        return 0.0
    vals = values[-window:]
    _, std = _ema(vals)
    return float(std * factor)


def _adaptive_synergy_threshold(
    history: list[dict[str, float]],
    window: int,
    *,
    factor: float = 2.0,
    weight: float = 1.0,
    confidence: float = 0.95,
    predictions: list[dict[str, float]] | None = None,
) -> float:
    """Return adaptive threshold for synergy metrics using weighted EMA.

    ``window`` specifies how many recent entries to include. ``factor`` scales
    the resulting bound. ``weight`` controls exponential weighting of newer
    samples (``1.0`` means no weighting). ``predictions`` should contain the
    corresponding predicted synergy values (with the same metric names) when
    available. ``confidence`` is retained for backward compatibility but is no
    longer used.
    """

    recent = history[-window:]
    pred = predictions[-window:] if predictions else []

    diffs: list[float] = []
    for idx, entry in enumerate(recent):
        pred_entry = pred[idx] if idx < len(pred) else {}
        for k, v in entry.items():
            if not k.startswith("synergy_"):
                continue
            if k.startswith("pred_"):
                continue
            pred_val = pred_entry.get(k)
            if pred_val is None:
                pred_val = entry.get(f"pred_{k}")
            if pred_val is not None:
                diffs.append(float(v) - float(pred_val))
            else:
                diffs.append(float(v))

    if not diffs:
        return 0.0

    if weight != 1.0:
        w = [weight ** i for i in range(len(diffs) - 1, -1, -1)]
        ema = sum(d * w_i for d, w_i in zip(diffs, w)) / sum(w)
        var = sum(w_i * (d - ema) ** 2 for d, w_i in zip(diffs, w)) / sum(w)
    else:
        ema = sum(diffs) / len(diffs)
        var = sum((d - ema) ** 2 for d in diffs) / len(diffs)

    std = var ** 0.5
    return float(std * factor)


def _diminishing_modules(
    history: dict[str, list[float]],
    flagged: set[str],
    threshold: float,
    consecutive: int = 3,
    std_threshold: float = 1e-3,
    *,
    confidence: float = 0.95,
) -> tuple[list[str], dict[str, float]]:
    """Return modules with ROI deltas that consistently fall within ``threshold``.

    The last ``consecutive`` deltas are evaluated using an exponential moving
    average and variance. A module is flagged when the EMA magnitude is below the
    given ``threshold`` and the exponentially weighted standard deviation is less
    than ``std_threshold``.
    """

    flags: list[str] = []
    confidences: dict[str, float] = {}
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


def _synergy_converged(
    history: list[dict[str, float]],
    window: int,
    threshold: float,
    std_threshold: float = 1e-3,
    ma_window: int | None = None,
    *,
    confidence: float = 0.95,
    stationarity_confidence: float = 0.95,
) -> tuple[bool, float, float]:
    """Return whether synergy metrics have converged.

    The check now also considers the correlation between the metric values and
    their sequence index and computes the confidence using the t-distribution
    when possible.
    """
    if len(history) < window:
        return False, 0.0, 0.0
    metrics: dict[str, list[float]] = {}
    for entry in history[-window:]:
        for k, v in entry.items():
            metrics.setdefault(k, []).append(float(v))
    max_abs = 0.0
    min_conf = 1.0
    ma_window = ma_window or window
    for vals in metrics.values():
        series = vals[-window:]
        ema, std = _ema(series[-ma_window:])
        n = len(series[-ma_window:])
        max_abs = max(max_abs, abs(ema))
        if std == 0:
            conf = 1.0 if abs(ema) <= threshold else 0.0
        else:
            se = std / math.sqrt(n)
            t_stat = abs(ema) / se
            p = 2 * (1 - t.cdf(t_stat, n - 1))
            conf = 1 - p
        stat_conf = conf
        try:  # pragma: no cover - optional dependency
            from statsmodels.tsa.stattools import adfuller

            ma_series: list[float] = []
            for i in range(len(series)):
                start = max(0, i - ma_window + 1)
                window_vals = series[start : i + 1]
                ma_series.append(sum(window_vals) / len(window_vals))
            adf_p = adfuller(ma_series)[1]
            stat_conf = 1.0 - float(adf_p)
        except Exception:
            pass

        # rolling correlation of metric values against iteration index
        roll_corr = 0.0
        if n > 1:
            corrs: list[float] = []
            size = min(ma_window, n)
            for i in range(n - size + 1):
                sub = series[i : i + size]
                if len(set(sub)) <= 1:
                    corrs.append(0.0)
                else:
                    idx = list(range(size))
                    c, _ = pearsonr(idx, sub)
                    corrs.append(c)
            if corrs:
                roll_corr = max(abs(c) for c in corrs)

        # combine EMA trend and variance confidence
        var_factor = 1.0
        if std_threshold > 0:
            var_factor = 1.0 / (1.0 + std / std_threshold)
        conf *= var_factor * (1.0 - abs(roll_corr))
        min_conf = min(min_conf, conf, stat_conf)

        if (
            abs(ema) > threshold
            or std > std_threshold
            or conf < confidence
            or stat_conf < stationarity_confidence
            or roll_corr > 0.3
        ):
            return False, max_abs, min_conf
    return True, max_abs, min_conf


def full_autonomous_run(args: argparse.Namespace) -> None:
    """Execute sandbox cycles until all modules show diminishing returns."""
    if getattr(args, "dashboard_port", None):
        history_file = Path(args.sandbox_data_dir or "sandbox_data") / "roi_history.json"
        dash = MetricsDashboard(str(history_file))
        Thread(target=dash.run, kwargs={"port": args.dashboard_port}, daemon=True).start()

    module_history: dict[str, list[float]] = {}
    flagged: set[str] = set()
    synergy_history: list[dict[str, float]] = []
    synergy_pred_history: list[dict[str, float]] = []
    roi_ma_history: list[float] = []
    synergy_ma_history: list[dict[str, float]] = []
    roi_threshold = getattr(args, "roi_threshold", None)
    env_val = os.getenv("ROI_THRESHOLD")
    if roi_threshold is None and env_val is not None:
        try:
            roi_threshold = float(env_val)
        except Exception:
            roi_threshold = None
    synergy_threshold = getattr(args, "synergy_threshold", None)
    env_val = os.getenv("SYNERGY_THRESHOLD")
    if synergy_threshold is None and env_val is not None:
        try:
            synergy_threshold = float(env_val)
        except Exception:
            synergy_threshold = None
    roi_confidence = getattr(args, "roi_confidence", None)
    env_val = os.getenv("ROI_CONFIDENCE")
    if roi_confidence is None and env_val is not None:
        try:
            roi_confidence = float(env_val)
        except Exception:
            roi_confidence = None
    synergy_confidence = getattr(args, "synergy_confidence", None)
    env_val = os.getenv("SYNERGY_CONFIDENCE")
    if synergy_confidence is None and env_val is not None:
        try:
            synergy_confidence = float(env_val)
        except Exception:
            synergy_confidence = None
    synergy_threshold_window = getattr(args, "synergy_threshold_window", None)
    env_val = os.getenv("SYNERGY_THRESHOLD_WINDOW")
    if synergy_threshold_window is None and env_val is not None:
        try:
            synergy_threshold_window = int(env_val)
        except Exception:
            synergy_threshold_window = None
    synergy_threshold_weight = getattr(args, "synergy_threshold_weight", None)
    env_val = os.getenv("SYNERGY_THRESHOLD_WEIGHT")
    if synergy_threshold_weight is None and env_val is not None:
        try:
            synergy_threshold_weight = float(env_val)
        except Exception:
            synergy_threshold_weight = None
    synergy_ma_window = getattr(args, "synergy_ma_window", None)
    env_val = os.getenv("SYNERGY_MA_WINDOW")
    if synergy_ma_window is None and env_val is not None:
        try:
            synergy_ma_window = int(env_val)
        except Exception:
            synergy_ma_window = None
    synergy_stationarity_confidence = getattr(args, "synergy_stationarity_confidence", None)
    env_val = os.getenv("SYNERGY_STATIONARITY_CONFIDENCE")
    if synergy_stationarity_confidence is None and env_val is not None:
        try:
            synergy_stationarity_confidence = float(env_val)
        except Exception:
            synergy_stationarity_confidence = None
    last_tracker = None
    iteration = 0
    roi_cycles = getattr(args, "roi_cycles", 3)
    synergy_cycles = getattr(args, "synergy_cycles", 3)
    if synergy_threshold_window is None:
        synergy_threshold_window = synergy_cycles
    if synergy_threshold_weight is None:
        synergy_threshold_weight = 1.0
    if synergy_ma_window is None:
        synergy_ma_window = synergy_cycles
    if synergy_stationarity_confidence is None:
        synergy_stationarity_confidence = synergy_confidence or 0.95

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
            pred_vals = {
                k[len("pred_"):]: v[-1]
                for k, v in tracker.metrics_history.items()
                if k.startswith("pred_synergy_") and v
            }
            if syn_vals:
                synergy_history.append(syn_vals)
                synergy_pred_history.append(pred_vals)
                # record moving average for synergy metrics
                ma_entry: dict[str, float] = {}
                for k in syn_vals:
                    vals = [h.get(k, 0.0) for h in synergy_history[-synergy_cycles:]]
                    ema, _ = _ema(vals) if vals else (0.0, 0.0)
                    ma_entry[k] = ema
                synergy_ma_history.append(ma_entry)
            history = getattr(tracker, "roi_history", [])
            if history:
                ema, _ = _ema(history[-roi_cycles:])
                roi_ma_history.append(ema)
        if last_tracker:
            if getattr(args, "auto_thresholds", False):
                roi_threshold = _adaptive_threshold(
                    last_tracker.roi_history, roi_cycles
                )
            elif roi_threshold is None:
                roi_threshold = last_tracker.diminishing()
            new_flags, _ = _diminishing_modules(
                module_history,
                flagged,
                roi_threshold,
                consecutive=roi_cycles,
                confidence=roi_confidence or 0.95,
            )
            flagged.update(new_flags)
        if last_tracker and getattr(args, "auto_thresholds", False):
            synergy_threshold = _adaptive_synergy_threshold(
                synergy_history,
                synergy_threshold_window,
                weight=synergy_threshold_weight,
                confidence=synergy_confidence or 0.95,
                predictions=synergy_pred_history,
            )
        elif last_tracker and synergy_threshold is None:
            synergy_threshold = last_tracker.diminishing()
        if synergy_threshold is not None:
            converged, ema_val, _ = _synergy_converged(
                synergy_history,
                synergy_cycles,
                synergy_threshold,
                ma_window=synergy_ma_window,
                confidence=synergy_confidence or 0.95,
                stationarity_confidence=synergy_stationarity_confidence or (synergy_confidence or 0.95),
            )
        else:
            converged, ema_val, _ = False, 0.0, 0.0
        if module_history and set(module_history) <= flagged and converged:
            logger.info(
                "synergy convergence reached",
                extra={"iteration": iteration, "ema": ema_val},
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
        "--no-preset-adapt",
        action="store_true",
        help="disable automatic environment preset adaptation",
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
        "--roi-threshold",
        type=float,
        help="override ROI delta threshold",
    )
    p_autorun.add_argument(
        "--roi-confidence",
        type=float,
        help="confidence level for ROI convergence",
    )
    p_autorun.add_argument(
        "--synergy-cycles",
        type=int,
        default=3,
        help="cycles below threshold before synergy convergence",
    )
    p_autorun.add_argument(
        "--synergy-threshold",
        type=float,
        help="override synergy threshold",
    )
    p_autorun.add_argument(
        "--synergy-threshold-window",
        type=int,
        help="number of recent synergy values/predictions used for the EMA",
    )
    p_autorun.add_argument(
        "--synergy-threshold-weight",
        type=float,
        help="exponential weight applied to newer synergy samples",
    )
    p_autorun.add_argument(
        "--synergy-confidence",
        type=float,
        help="confidence level for synergy convergence",
    )
    p_autorun.add_argument(
        "--synergy-ma-window",
        type=int,
        help="window size for synergy moving average",
    )
    p_autorun.add_argument(
        "--synergy-stationarity-confidence",
        type=float,
        help="confidence level for synergy stationarity test",
    )
    p_autorun.add_argument(
        "--auto-thresholds",
        action="store_true",
        help="compute convergence thresholds adaptively",
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
        "--roi-threshold",
        type=float,
        help="override ROI delta threshold",
    )
    p_complete.add_argument(
        "--roi-confidence",
        type=float,
        help="confidence level for ROI convergence",
    )
    p_complete.add_argument(
        "--synergy-cycles",
        type=int,
        default=3,
        help="cycles below threshold before synergy convergence",
    )
    p_complete.add_argument(
        "--synergy-threshold",
        type=float,
        help="override synergy threshold",
    )
    p_complete.add_argument(
        "--synergy-threshold-window",
        type=int,
        help="number of recent synergy values/predictions used for the EMA",
    )
    p_complete.add_argument(
        "--synergy-threshold-weight",
        type=float,
        help="exponential weight applied to newer synergy samples",
    )
    p_complete.add_argument(
        "--synergy-confidence",
        type=float,
        help="confidence level for synergy convergence",
    )
    p_complete.add_argument(
        "--synergy-ma-window",
        type=int,
        help="window size for synergy moving average",
    )
    p_complete.add_argument(
        "--synergy-stationarity-confidence",
        type=float,
        help="confidence level for synergy stationarity test",
    )
    p_complete.add_argument(
        "--auto-thresholds",
        action="store_true",
        help="compute convergence thresholds adaptively",
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

