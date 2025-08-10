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
import platform
import subprocess
from sandbox_settings import SandboxSettings
try:  # optional dependency
    from scipy.stats import pearsonr, t, levene
except Exception:  # pragma: no cover - fallback when scipy is missing
    import types

    def pearsonr(x, y):  # type: ignore
        n = len(x)
        if n == 0:
            return 0.0, 0.0
        mx = sum(x) / n
        my = sum(y) / n
        num = sum((a - mx) * (b - my) for a, b in zip(x, y))
        den_x = sum((a - mx) ** 2 for a in x)
        den_y = sum((b - my) ** 2 for b in y)
        denom = math.sqrt(den_x * den_y)
        if denom == 0:
            return 0.0, 0.0
        return num / denom, 0.0

    class _T:
        @staticmethod
        def cdf(val: float, df: float) -> float:
            return 0.5 * (1.0 + math.erf(val / math.sqrt(2.0)))

    t = _T()

    def levene(*a, **k):  # type: ignore
        return types.SimpleNamespace(pvalue=1.0)
from threading import Thread

from menace.metrics_dashboard import MetricsDashboard
from logging_utils import get_logger, setup_logging

from .environment import SANDBOX_ENV_PRESETS, simulate_full_environment

try:  # optional import for tests
    from menace.environment_generator import generate_presets
except Exception:  # pragma: no cover
    generate_presets = lambda n=None: [{}]  # type: ignore

logger = get_logger(__name__)


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
            summary.update(
                0.0,
                delta,
                modules=[f"preset_{idx}"],
                metrics={"security_score": sec_val},
            )
        logger.info("sandbox presets complete", extra={"ranking": summary.rankings()})
        return

    preset = presets[0]
    sandbox_main(preset, args)
    if getattr(args, "coverage_report", False):
        from .environment import coverage_summary

        print(json.dumps(coverage_summary(), indent=2))


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
        logger.info(
            "%s ROI=%.3f security_score=%.3f",
            name,
            roi_val,
            sec_val,
            extra={
                "preset": name,
                "roi": roi_val,
                "security_score": sec_val,
            },
        )


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
            logger.info(
                "%s %s=%.3f",
                name,
                metric,
                val,
                extra={"preset": name, "metric": metric, "value": val},
            )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def synergy_metrics(file: str, *, window: int = 5, plot: bool = False) -> None:
    """Print or plot the latest synergy metrics from ``file``.

    The ``file`` argument should point to ``roi_history.json`` generated by
    sandbox runs.  ``window`` controls the exponential moving average window
    applied to each metric.  When ``plot`` is ``True`` and ``matplotlib`` is
    installed a bar chart showing the latest values and moving averages is
    displayed.
    """

    from menace.roi_tracker import ROITracker

    tracker = ROITracker()
    try:
        tracker.load_history(file)
    except Exception:
        logger.exception("failed to load history %s", file)
        return

    synergy_names = {
        n
        for n in tracker.metrics_history
        if n.startswith("synergy_")
    } | set(tracker.synergy_metrics_history)

    data: list[tuple[str, float, float]] = []
    for name in sorted(synergy_names):
        hist = tracker.synergy_metrics_history.get(name)
        if hist is None:
            hist = tracker.metrics_history.get(name, [])
        last = float(hist[-1]) if hist else 0.0
        ma = _ema(hist[-window:])[0] if hist else 0.0
        data.append((name, last, ma))

    if plot:
        try:  # pragma: no cover - optional dependency
            import matplotlib.pyplot as plt  # type: ignore

            labels = [d[0] for d in data]
            last_vals = [d[1] for d in data]
            ma_vals = [d[2] for d in data]
            x = range(len(labels))
            width = 0.35
            fig, ax = plt.subplots()
            ax.bar([i - width / 2 for i in x], last_vals, width, label="last")
            ax.bar([i + width / 2 for i in x], ma_vals, width, label=f"EMA{window}")
            ax.set_xticks(list(x))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_ylabel("value")
            ax.legend()
            fig.tight_layout()
            plt.show()
        except Exception:
            logger.exception("failed to plot synergy metrics")
            plot = False

    if not plot:
        for name, last, ma in data:
            logger.info(
                "%s last=%.3f ema=%.3f",
                name,
                last,
                ma,
                extra={"metric": name, "last": last, "ema": ma},
            )


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
        ema_sq = alpha * (v**2) + (1 - alpha) * ema_sq
    var = ema_sq - ema**2
    if var < 1e-12:
        var = 0.0
    return ema, var**0.5


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
    corresponding predicted synergy values with the same metric names when
    available. ``confidence`` is retained for backward compatibility but is no
    longer used.

    The bound is derived from the weighted standard deviation of
    ``actual - predicted`` values.  When a ``weight`` other than ``1.0`` is
    provided the weights ``weight**i`` are applied from newest to oldest
    samples.  The exponential moving average (EMA) is calculated as::

        ema = sum(w_i * x_i) / sum(w_i)

    and the variance as ``sum(w_i * (x_i - ema)**2) / sum(w_i)`` where
    ``x_i`` are the differences.  The square root of this variance multiplied by
    ``factor`` becomes the adaptive threshold.  This explicit formula is
    recorded here for maintainability as :func:`_ema` mirrors these
    calculations.
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
        w = [weight**i for i in range(len(diffs) - 1, -1, -1)]
        ema = sum(d * w_i for d, w_i in zip(diffs, w)) / sum(w)
        var = sum(w_i * (d - ema) ** 2 for d, w_i in zip(diffs, w)) / sum(w)
    else:
        ema = sum(diffs) / len(diffs)
        var = sum((d - ema) ** 2 for d in diffs) / len(diffs)

    std = var**0.5
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
    variance_confidence: float = 0.95,
) -> tuple[bool, float, float]:
    """Return whether synergy metrics have converged.

    The check now also considers the correlation between the metric values and
    their sequence index and computes the confidence using the t-distribution
    when possible.  If ``statsmodels`` and ``scipy`` are installed the
    Augmented Dickey-Fuller test is applied over a rolling window along with a
    Levene variance test.  When these dependencies are missing a lightweight
    AR(1) residual estimate is used as a fallback so that convergence can still
    be estimated.
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
        ma_series: list[float] = []
        for i in range(len(series)):
            start = max(0, i - ma_window + 1)
            window_vals = series[start : i + 1]
            ma_series.append(sum(window_vals) / len(window_vals))
        try:  # pragma: no cover - optional dependency
            from statsmodels.tsa.stattools import adfuller

            adf_ps: list[float] = []
            size = min(len(ma_series), ma_window)
            if size >= 3:
                for j in range(len(ma_series) - size + 1):
                    p_val = adfuller(ma_series[j : j + size])[1]
                    adf_ps.append(float(p_val))
            if adf_ps:
                stat_conf = 1.0 - max(adf_ps)
        except Exception:
            # fallback when statsmodels is unavailable: AR(1) coefficient check
            if len(ma_series) > 2:
                x = ma_series[:-1]
                y = ma_series[1:]
                denom = sum(v * v for v in x)
                if denom > 0:
                    phi = sum(a * b for a, b in zip(x, y)) / denom
                    stat_conf = max(0.0, 1.0 - min(1.0, abs(1.0 - phi)))

        var_change_conf = 0.0
        try:  # pragma: no cover - optional dependency
            from scipy.stats import levene

            half = n // 2
            if half > 1 and n - half > 1:
                lev_p = levene(series[:half], series[half:]).pvalue
                var_change_conf = 1.0 - float(lev_p)
        except Exception:
            # fallback when scipy is unavailable: estimate variance change by
            # comparing sample variances of the first and second halves
            half = n // 2
            if half > 1 and n - half > 1:
                m1 = sum(series[:half]) / half
                m2 = sum(series[half:]) / (n - half)
                v1 = sum((x - m1) ** 2 for x in series[:half]) / (half - 1)
                v2 = sum((x - m2) ** 2 for x in series[half:]) / (n - half - 1)
                avg_var = (v1 + v2) / 2
                if avg_var > 0:
                    ratio = abs(v1 - v2) / avg_var
                    var_change_conf = min(1.0, ratio)
                else:
                    var_change_conf = 0.0
        if math.isnan(var_change_conf):
            var_change_conf = 1.0

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

        # combine ADF, variance and correlation into weighted confidence
        var_factor = 1.0
        if std_threshold > 0:
            var_factor = 1.0 / (1.0 + (std / std_threshold) ** 2)
        trend_conf = conf * var_factor
        combined = (
            0.6 * stat_conf
            + 0.3 * trend_conf
            + 0.07 * (1.0 - var_change_conf)
            + 0.03 * (1.0 - abs(roll_corr))
        )
        if math.isnan(combined):
            combined = 0.0
        min_conf = min(min_conf, combined)

        if (
            abs(ema) > threshold
            or std > std_threshold
            or combined < confidence
            or stat_conf < stationarity_confidence
            or var_change_conf >= variance_confidence
            or roll_corr > 0.3
        ):
            return False, max_abs, min_conf
    return True, max_abs, min_conf


def adaptive_synergy_convergence(
    history: list[dict[str, float]],
    window: int,
    *,
    threshold: float | None = None,
    threshold_window: int | None = None,
    factor: float = 2.0,
    weight: float = 1.0,
    confidence: float = 0.95,
) -> tuple[bool, float, float]:
    """Return synergy convergence using EWMA and t-test statistics.

    The metrics of the last ``window`` iterations are examined.  When
    ``threshold`` is ``None`` the bound is calculated from the most recent
    ``threshold_window`` samples via :func:`_adaptive_synergy_threshold` using a
    weighted EMA.  For each synergy metric the EMA and standard deviation are
    computed with :func:`_ema` and the mean ``ema`` is compared against the
    threshold by a two-sided t-test::

        se = std / sqrt(n)
        t_stat = |ema| / se
        p = 2 * (1 - cdf(t_stat, df=n-1))

    where ``cdf`` is the cumulative density function of the Student's
    t-distribution.  ``confidence`` is ``1 - p`` and ``n`` is the number of
    samples.  Convergence is reached only if ``|ema|`` is below the threshold
    and the confidence for all metrics is greater than the supplied
    ``confidence`` value.  The function returns ``(converged, max_abs_ema,
    min_confidence)`` where ``min_confidence`` is the lowest t-test confidence
    found.
    """

    if threshold_window is None:
        threshold_window = window

    thr = threshold
    if thr is None:
        thr = _adaptive_synergy_threshold(
            history, threshold_window, factor=factor, weight=weight
        )

    if len(history) < window:
        return False, 0.0, 0.0

    metrics: dict[str, list[float]] = {}
    for entry in history[-window:]:
        for k, v in entry.items():
            if k.startswith("synergy_"):
                metrics.setdefault(k, []).append(float(v))

    max_abs = 0.0
    min_conf = 1.0
    for vals in metrics.values():
        ema, std = _ema(vals)
        max_abs = max(max_abs, abs(ema))
        n = len(vals)
        if n < 2 or std == 0:
            conf = 1.0 if abs(ema) <= thr else 0.0
        else:
            se = std / math.sqrt(n)
            t_stat = abs(ema) / se
            p = 2 * (1 - t.cdf(t_stat, n - 1))
            conf = 1 - p
        min_conf = min(min_conf, conf)
        if abs(ema) > thr or conf < confidence:
            return False, max_abs, min_conf

    return True, max_abs, min_conf


def full_autonomous_run(
    args: argparse.Namespace,
    *,
    synergy_history: list[dict[str, float]] | None = None,
    synergy_ma_history: list[dict[str, float]] | None = None,
) -> None:
    """Execute sandbox cycles until all modules show diminishing returns."""
    if getattr(args, "dashboard_port", None):
        history_file = (
            Path(args.sandbox_data_dir or "sandbox_data") / "roi_history.json"
        )
        dash = MetricsDashboard(str(history_file))
        Thread(
            target=dash.run, kwargs={"port": args.dashboard_port}, daemon=True
        ).start()

    module_history: dict[str, list[float]] = {}
    flagged: set[str] = set()
    synergy_history = list(synergy_history or [])
    synergy_pred_history: list[dict[str, float]] = []
    roi_ma_history: list[float] = []
    synergy_ma_history = list(synergy_ma_history or [])
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
    synergy_stationarity_confidence = getattr(
        args, "synergy_stationarity_confidence", None
    )
    env_val = os.getenv("SYNERGY_STATIONARITY_CONFIDENCE")
    if synergy_stationarity_confidence is None and env_val is not None:
        try:
            synergy_stationarity_confidence = float(env_val)
        except Exception:
            synergy_stationarity_confidence = None
    synergy_std_threshold = getattr(args, "synergy_std_threshold", None)
    env_val = os.getenv("SYNERGY_STD_THRESHOLD")
    if synergy_std_threshold is None and env_val is not None:
        try:
            synergy_std_threshold = float(env_val)
        except Exception:
            synergy_std_threshold = None
    synergy_variance_confidence = getattr(args, "synergy_variance_confidence", None)
    env_val = os.getenv("SYNERGY_VARIANCE_CONFIDENCE")
    if synergy_variance_confidence is None and env_val is not None:
        try:
            synergy_variance_confidence = float(env_val)
        except Exception:
            synergy_variance_confidence = None
    last_tracker = None
    iteration = 0
    roi_cycles = getattr(args, "roi_cycles", 3)
    env_val = os.getenv("ROI_CYCLES")
    if env_val is not None:
        try:
            roi_cycles = int(env_val)
        except Exception:
            pass
    synergy_cycles = getattr(args, "synergy_cycles", 3)
    env_val = os.getenv("SYNERGY_CYCLES")
    if env_val is not None:
        try:
            synergy_cycles = int(env_val)
        except Exception:
            pass
    if synergy_threshold_window is None:
        synergy_threshold_window = synergy_cycles
    if synergy_threshold_weight is None:
        synergy_threshold_weight = 1.0
    if synergy_ma_window is None:
        synergy_ma_window = synergy_cycles
    if synergy_stationarity_confidence is None:
        synergy_stationarity_confidence = synergy_confidence or 0.95
    if synergy_std_threshold is None:
        synergy_std_threshold = 1e-3
    if synergy_variance_confidence is None:
        synergy_variance_confidence = synergy_confidence or 0.95

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
                include_orphans=getattr(args, "include_orphans", None),
                discover_orphans=getattr(args, "discover_orphans", None),
                discover_isolated=getattr(args, "discover_isolated", None),
                recursive_orphans=getattr(args, "recursive_orphans", None),
                recursive_isolated=getattr(args, "recursive_isolated", None),
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
                k[len("pred_") :]: v[-1]
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
                win = min(len(last_tracker.roi_history), roi_cycles)
                roi_threshold = _adaptive_threshold(
                    last_tracker.roi_history,
                    win,
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
            win = min(len(synergy_history), synergy_threshold_window)
            synergy_threshold = _adaptive_synergy_threshold(
                synergy_history,
                win,
                weight=synergy_threshold_weight,
                confidence=synergy_confidence or 0.95,
                predictions=synergy_pred_history,
            )
            synergy_ma_window = max(1, win)
        elif last_tracker and synergy_threshold is None:
            synergy_threshold = last_tracker.diminishing()
        if synergy_threshold is not None:
            converged, ema_val, _ = _synergy_converged(
                synergy_history,
                synergy_cycles,
                synergy_threshold,
                std_threshold=synergy_std_threshold,
                ma_window=synergy_ma_window,
                confidence=synergy_confidence or 0.95,
                stationarity_confidence=synergy_stationarity_confidence
                or (synergy_confidence or 0.95),
                variance_confidence=synergy_variance_confidence
                or (synergy_confidence or 0.95),
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
        logger.info("=== Final Module Rankings ===", extra={"iteration": iteration})
        for mod, total in last_tracker.rankings():
            logger.info(
                "%s: %.3f",
                mod,
                total,
                extra={"iteration": iteration, "module": mod, "total": total},
            )
        logger.info("=== Metrics ===", extra={"iteration": iteration})
        for name, vals in last_tracker.metrics_history.items():
            if vals:
                logger.info(
                    "%s: %.3f",
                    name,
                    vals[-1],
                    extra={"iteration": iteration, "metric": name, "value": vals[-1]},
                )
    else:
        logger.warning("No sandbox runs executed", extra={"iteration": iteration})


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


def cleanup_command() -> None:
    """Purge stale resources and retry previous cleanup failures."""

    from sandbox_runner.environment import (
        purge_leftovers,
        retry_failed_cleanup,
        _PURGE_FILE_LOCK,
    )

    purge_leftovers()
    with _PURGE_FILE_LOCK:
        retry_failed_cleanup()


def check_resources_command() -> None:
    """Purge leftovers and retry cleanup then print a summary."""

    from sandbox_runner import environment as env

    before_containers = env._STALE_CONTAINERS_REMOVED
    before_overlays = env._STALE_VMS_REMOVED
    env.purge_leftovers()
    with env._PURGE_FILE_LOCK:
        env.retry_failed_cleanup()

    removed_containers = env._STALE_CONTAINERS_REMOVED - before_containers
    removed_overlays = env._STALE_VMS_REMOVED - before_overlays
    logger.info(
        "Removed %s containers and %s overlays",
        removed_containers,
        removed_overlays,
        extra={
            "removed_containers": removed_containers,
            "removed_overlays": removed_overlays,
        },
    )


def install_autopurge_command() -> None:
    """Install scheduled cleanup using systemd or Windows Task Scheduler."""

    base = Path(__file__).resolve().parent.parent
    system = platform.system().lower()

    if system == "windows":
        xml = base / "systemd" / "windows_sandbox_purge.xml"
        try:
            subprocess.run(
                [
                    "schtasks",
                    "/Create",
                    "/TN",
                    "SandboxPurge",
                    "/XML",
                    str(xml),
                    "/F",
                ],
                check=True,
            )
            logger.info(
                "Scheduled task installed. Adjust via Task Scheduler if needed.",
                extra={"system": system},
            )
        except Exception:
            logger.warning(
                "Failed to import scheduled task. Run the following manually:",
                extra={"system": system},
            )
            logger.warning(
                "schtasks /Create /TN SandboxPurge /XML %s /F",
                xml,
                extra={"system": system},
            )
        return

    if system in {"linux", "darwin"} and shutil.which("systemctl"):
        service = base / "systemd" / "sandbox_autopurge.service"
        timer = base / "systemd" / "sandbox_autopurge.timer"
        try:
            if hasattr(os, "geteuid") and os.geteuid() == 0:
                unit_dir = Path("/etc/systemd/system")
                user_flag: list[str] = []
            else:
                unit_dir = Path.home() / ".config" / "systemd" / "user"
                unit_dir.mkdir(parents=True, exist_ok=True)
                user_flag = ["--user"]
            shutil.copy(service, unit_dir / service.name)
            shutil.copy(timer, unit_dir / timer.name)
            subprocess.run(["systemctl", *user_flag, "daemon-reload"], check=True)
            subprocess.run(
                ["systemctl", *user_flag, "enable", "--now", "sandbox_autopurge.timer"],
                check=True,
            )
            logger.info(
                "sandbox_autopurge timer installed and enabled",
                extra={"system": system, "unit_dir": str(unit_dir)},
            )
        except Exception:
            logger.warning(
                "Failed to install systemd units automatically.",
                extra={"system": system, "unit_dir": str(unit_dir)},
            )
            logger.warning(
                "Copy %s and %s to %s and enable with systemctl",
                service,
                timer,
                unit_dir,
                extra={"system": system, "unit_dir": str(unit_dir)},
            )
        return

    logger.warning(
        "Automatic installation not supported on this platform.",
        extra={"system": system},
    )
    logger.warning(
        "See systemd/sandbox_autopurge.service for manual instructions.",
        extra={"system": system},
    )


def main(argv: List[str] | None = None) -> None:
    """Entry point for command line execution."""
    parser = argparse.ArgumentParser(description="Run Menace sandbox")
    parser.add_argument(
        "--workflow-sim",
        action="store_true",
        help="simulate workflows instead of repo sections",
    )
    parser.add_argument(
        "--workflow-db", default="workflows.db", help="path to workflow database"
    )
    parser.add_argument(
        "--dynamic-workflows",
        action="store_true",
        help="generate workflows from module groups when missing",
    )
    parser.add_argument(
        "--sandbox-data-dir", help="override data directory for sandbox mode"
    )
    parser.add_argument(
        "--autodiscover-modules",
        action="store_true",
        help="generate module map automatically (or set SANDBOX_AUTO_MAP=1)",
    )
    parser.add_argument(
        "--module-algorithm",
        choices=["greedy", "label", "hdbscan"],
        default=os.getenv("SANDBOX_MODULE_ALGO"),
        help="module clustering algorithm",
    )
    parser.add_argument(
        "--module-threshold",
        type=float,
        default=os.getenv("SANDBOX_MODULE_THRESHOLD"),
        help="semantic similarity threshold",
    )
    parser.add_argument(
        "--module-semantic",
        action="store_true",
        help="enable docstring similarity (or SANDBOX_SEMANTIC_MODULES=1)",
    )
    parser.add_argument(
        "--refresh-module-map",
        action="store_true",
        help="rebuild module map before sandbox cycles",
    )
    parser.add_argument(
        "--preset-count",
        type=int,
        help="number of presets to generate when none are provided",
    )
    parser.add_argument(
        "--max-prompt-length", type=int, help="maximum characters for GPT prompts"
    )
    parser.add_argument(
        "--summary-depth", type=int, help="lines to keep when summarising snippets"
    )
    parser.add_argument(
        "--max-recursion-depth",
        type=int,
        default=os.getenv("SANDBOX_MAX_RECURSION_DEPTH"),
        help="maximum depth when resolving dependency chains",
    )
    parser.add_argument(
        "--discover-orphans",
        action="store_false",
        dest="discover_orphans",
        default=None,
        help="disable automatic find_orphan_modules scans",
    )
    parser.add_argument(
        "--discover-isolated",
        action="store_true",
        dest="discover_isolated",
        default=None,
        help="automatically run discover_isolated_modules before the orphan scan",
    )
    parser.add_argument(
        "--no-discover-isolated",
        action="store_false",
        dest="discover_isolated",
        help="disable discover_isolated_modules before the orphan scan",
    )
    parser.add_argument(
        "--recursive-isolated",
        action="store_true",
        dest="recursive_isolated",
        default=None,
        help="recurse through dependencies of isolated modules (default: enabled)",
    )
    parser.add_argument(
        "--no-recursive-isolated",
        action="store_false",
        dest="recursive_isolated",
        help="disable recursion when including isolated modules",
    )
    parser.add_argument(
        "--auto-include-isolated",
        action="store_true",
        help=(
            "automatically include isolated modules recursively (sets "
            "SANDBOX_AUTO_INCLUDE_ISOLATED=1 and SANDBOX_RECURSIVE_ISOLATED=1)"
        ),
    )
    parser.add_argument(
        "--recursive-orphans",
        "--recursive-include",
        action="store_true",
        dest="recursive_orphans",
        default=None,
        help=(
            "recursively integrate orphan dependency chains (sets "
            "SANDBOX_RECURSIVE_ORPHANS=1; alias: --recursive-include)"
        ),
    )
    parser.add_argument(
        "--no-recursive-orphans",
        "--no-recursive-include",
        action="store_false",
        dest="recursive_orphans",
        help=(
            "disable recursive orphan dependency integration (sets "
            "SANDBOX_RECURSIVE_ORPHANS=0)"
        ),
    )
    parser.add_argument(
        "--include-orphans",
        action="store_false",
        dest="include_orphans",
        default=None,
        help="skip modules listed in sandbox_data/orphan_modules.json",
    )
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
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="remove leftover containers and QEMU overlay files",
    )
    parser.add_argument(
        "--purge-stale",
        action="store_true",
        help="remove stale containers and VM overlays then exit",
    )
    parser.add_argument(
        "--misuse-stubs",
        action="store_true",
        help="append adversarial misuse input stubs",
    )
    parser.add_argument(
        "--coverage-report",
        action="store_true",
        help="print coverage summary after sandbox execution",
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

    p_metrics = sub.add_parser(
        "synergy-metrics", help="print or plot recent synergy metrics"
    )
    p_metrics.add_argument(
        "--file",
        default="sandbox_data/roi_history.json",
        help="path to roi_history.json",
    )
    p_metrics.add_argument(
        "--window",
        type=int,
        default=5,
        help="EMA window for moving average",
    )
    p_metrics.add_argument("--plot", action="store_true", help="show matplotlib plot")

    sub.add_parser(
        "cleanup",
        help="purge leftovers and retry previously failed cleanup",
    )

    sub.add_parser(
        "check-resources",
        help="purge leftovers, retry cleanup and report removed resources",
    )

    sub.add_parser(
        "install-autopurge",
        help="install scheduled cleanup service (bootstrap runs this automatically)",
    )

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
        "--synergy-std-threshold",
        type=float,
        help="standard deviation threshold for synergy convergence",
    )
    p_autorun.add_argument(
        "--synergy-variance-confidence",
        type=float,
        help="confidence level for variance change test",
    )
    p_autorun.add_argument(
        "--auto-thresholds",
        action="store_true",
        help="adjust ROI and synergy thresholds automatically",
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
        "--synergy-std-threshold",
        type=float,
        help="standard deviation threshold for synergy convergence",
    )
    p_complete.add_argument(
        "--synergy-variance-confidence",
        type=float,
        help="confidence level for variance change test",
    )
    p_complete.add_argument(
        "--auto-thresholds",
        action="store_true",
        help="adjust ROI and synergy thresholds automatically",
    )

    args = parser.parse_args(argv)

    if getattr(args, "misuse_stubs", False):
        os.environ["SANDBOX_MISUSE_STUBS"] = "1"

    settings = SandboxSettings()
    auto_include_isolated = bool(getattr(settings, "auto_include_isolated", True) or getattr(args, "auto_include_isolated", False))
    recursive_orphans = getattr(settings, "recursive_orphan_scan", True)
    if args.recursive_orphans is not None:
        recursive_orphans = args.recursive_orphans
    recursive_isolated = getattr(settings, "recursive_isolated", True)
    if args.recursive_isolated is not None:
        recursive_isolated = args.recursive_isolated
    if auto_include_isolated:
        recursive_isolated = True

    args.auto_include_isolated = auto_include_isolated
    args.recursive_orphans = recursive_orphans
    args.recursive_isolated = recursive_isolated

    os.environ["SANDBOX_AUTO_INCLUDE_ISOLATED"] = "1" if auto_include_isolated else "0"
    os.environ["SELF_TEST_AUTO_INCLUDE_ISOLATED"] = "1" if auto_include_isolated else "0"
    val = "1" if recursive_orphans else "0"
    os.environ["SANDBOX_RECURSIVE_ORPHANS"] = val
    os.environ["SELF_TEST_RECURSIVE_ORPHANS"] = val
    val_iso = "1" if recursive_isolated else "0"
    os.environ["SANDBOX_RECURSIVE_ISOLATED"] = val_iso
    os.environ["SELF_TEST_RECURSIVE_ISOLATED"] = val_iso
    if auto_include_isolated:
        os.environ["SANDBOX_DISCOVER_ISOLATED"] = "1"
        os.environ["SELF_TEST_DISCOVER_ISOLATED"] = "1"

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
    if args.max_recursion_depth is not None:
        os.environ["SANDBOX_MAX_RECURSION_DEPTH"] = str(args.max_recursion_depth)

    if getattr(args, "purge_stale", False):
        from sandbox_runner.environment import purge_leftovers

        purge_leftovers()
        return

    if getattr(args, "cleanup", False):
        from sandbox_runner.environment import purge_leftovers

        purge_leftovers()
        return

    if getattr(args, "cmd", None) == "rank-scenarios":
        rank_scenarios(args.paths)
        return

    if getattr(args, "cmd", None) == "rank-synergy":
        from sandbox_runner.environment import aggregate_synergy_metrics

        results = aggregate_synergy_metrics(args.paths, args.metric)
        for name, val in results:
            logger.info(
                "%s %s=%.3f",
                name,
                args.metric,
                val,
                extra={"preset": name, "metric": args.metric, "value": val},
            )
        return

    if getattr(args, "cmd", None) == "rank-scenario-synergy":
        rank_scenario_synergy(args.paths, args.metric)
        return

    if getattr(args, "cmd", None) == "synergy-metrics":
        synergy_metrics(args.file, window=args.window, plot=getattr(args, "plot", False))
        return

    if getattr(args, "cmd", None) == "cleanup":
        cleanup_command()
        return

    if getattr(args, "cmd", None) == "check-resources":
        check_resources_command()
        return

    if getattr(args, "cmd", None) == "install-autopurge":
        install_autopurge_command()
        return

    if getattr(args, "cmd", None) == "full-autonomous-run":
        full_autonomous_run(args)
        return

    if getattr(args, "cmd", None) == "run-complete":
        run_complete(args)
        return

    if args.workflow_sim:
        from sandbox_runner.environment import run_workflow_simulations

        run_workflow_simulations(
            args.workflow_db,
            dynamic_workflows=args.dynamic_workflows,
            module_algorithm=args.module_algorithm or "greedy",
            module_threshold=float(args.module_threshold)
            if args.module_threshold is not None
            else 0.1,
            module_semantic=args.module_semantic,
        )
    else:
        _run_sandbox(args)


if __name__ == "__main__":  # pragma: no cover - CLI usage
    setup_logging()
    main()
