from __future__ import annotations

import argparse
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, List, Mapping
import tempfile
import shutil
import math
import platform
import subprocess
from sandbox_settings import SandboxSettings, load_sandbox_settings
from dynamic_path_router import resolve_path, path_for_prompt
from pydantic import ValidationError

try:
    from context_builder_util import create_context_builder
except ModuleNotFoundError:  # pragma: no cover - package-relative import fallback
    from import_compat import load_internal

    create_context_builder = load_internal("context_builder_util").create_context_builder
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

try:  # optional dependency
    from menace.metrics_dashboard import MetricsDashboard
    _METRICS_DASHBOARD_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - fallback when Flask is unavailable
    MetricsDashboard = None  # type: ignore[assignment]
    _METRICS_DASHBOARD_ERROR = exc
from logging_utils import get_logger, setup_logging, set_correlation_id

from foresight_tracker import ForesightTracker
try:  # optional dependency
    from .meta_logger import _SandboxMetaLogger
except Exception:  # pragma: no cover - best effort
    _SandboxMetaLogger = None  # type: ignore

try:  # optional import for tests
    from menace.environment_generator import generate_presets  # type: ignore
    _ENV_GEN_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover
    generate_presets = None  # type: ignore
    _ENV_GEN_ERROR = exc


def _fallback_presets(count: int | None = None) -> list[dict[str, Any]]:
    """Deterministic presets when :mod:`menace.environment_generator` is absent."""

    base = {
        "CPU_LIMIT": "1",
        "MEMORY_LIMIT": "512Mi",
        "DISK_LIMIT": "1Gi",
        "NETWORK_LATENCY_MS": 50,
        "BANDWIDTH_LIMIT": "10Mbps",
        "PACKET_LOSS": 0.0,
        "SECURITY_LEVEL": 1,
        "THREAT_INTENSITY": 10,
    }
    if not count or count < 1:
        count = 1
    return [base.copy() for _ in range(count)]


logger = get_logger(__name__)

_settings_cache: SandboxSettings | None = None
_settings_mtime: float | None = None
_settings_path: str | None = None
_settings_env_snapshot: dict[str, str | None] | None = None


def _build_env_alias_map() -> dict[str, tuple[str, ...]]:
    """Return mapping of settings field names to environment aliases."""

    aliases: dict[str, tuple[str, ...]] = {}
    for name, field in SandboxSettings.model_fields.items():
        extra = getattr(field, "json_schema_extra", None)
        if not isinstance(extra, dict):
            continue
        env_value = extra.get("env")
        if isinstance(env_value, str):
            aliases[name] = (env_value,)
        elif isinstance(env_value, (list, tuple, set)):
            filtered = tuple(str(item) for item in env_value if isinstance(item, str))
            if filtered:
                aliases[name] = filtered
    return aliases


_SETTINGS_ENV_ALIASES = _build_env_alias_map()
_SETTINGS_ENV_KEYS = {
    env_name
    for aliases in _SETTINGS_ENV_ALIASES.values()
    for env_name in aliases
}


def _clear_invalid_env_overrides(exc: ValidationError) -> bool:
    """Remove invalid environment overrides referenced in ``exc``.

    Returns ``True`` when at least one variable was removed which allows the
    caller to retry settings initialisation with a clean environment.
    """

    handled = False
    for error in exc.errors():
        loc = error.get("loc")
        if not loc:
            continue
        field_name = loc[0]
        if not isinstance(field_name, str):
            continue
        env_aliases = _SETTINGS_ENV_ALIASES.get(field_name, ())
        for env_name in env_aliases:
            if env_name not in os.environ:
                continue
            raw_value = os.environ.pop(env_name)
            logger.warning(
                "invalid %s value %r; ignoring environment override",
                env_name,
                raw_value,
            )
            handled = True
    return handled


def get_settings() -> SandboxSettings:
    """Return cached :class:`SandboxSettings`, reloading when the config changes."""
    global _settings_cache, _settings_mtime, _settings_path, _settings_env_snapshot
    path = os.getenv("SANDBOX_SETTINGS_PATH")
    mtime = None
    if path:
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = None
    current_snapshot = {key: os.environ.get(key) for key in _SETTINGS_ENV_KEYS}
    if (
        _settings_cache is None
        or path != _settings_path
        or mtime != _settings_mtime
        or current_snapshot != _settings_env_snapshot
    ):
        while True:
            try:
                if path:
                    _settings_cache = load_sandbox_settings(path)
                else:
                    _settings_cache = SandboxSettings()
            except ValidationError as exc:
                if not _clear_invalid_env_overrides(exc):
                    raise
                # One or more environment variables were removed. Retry the
                # settings load so the defaults can be applied.
                continue
            break
        _settings_path = path
        _settings_mtime = mtime
        _settings_env_snapshot = {key: os.environ.get(key) for key in _SETTINGS_ENV_KEYS}
    return _settings_cache


def _run_sandbox(args: argparse.Namespace, sandbox_main=None) -> None:
    """Execute sandbox runs for one or multiple environment presets."""
    if sandbox_main is None:
        from sandbox_runner import _sandbox_main as sandbox_main

    from .environment import load_presets, simulate_full_environment

    presets = load_presets()
    settings = get_settings()
    source = "SANDBOX_ENV_PRESETS" if presets != [{}] else "default"
    if presets == [{}] and not settings.sandbox_env_presets:
        if settings.sandbox_generate_presets:
            count = getattr(args, "preset_count", None)
            if generate_presets is not None:
                presets = generate_presets(count)  # type: ignore
                source = "menace.environment_generator"
            else:
                logger.warning(
                    "menace.environment_generator unavailable; using fallback presets",
                    extra={"error": str(_ENV_GEN_ERROR)},
                )
                presets = _fallback_presets(count)
                source = "fallback"
    if not presets or presets == [{}]:
        logger.error(
            "no valid presets available; install menace.environment_generator "
            "or set SANDBOX_ENV_PRESETS",
            extra={"preset_source": source},
        )
        raise SystemExit(1)
    logger.info(
        "loaded %d preset(s) from %s",
        len(presets),
        source,
        extra={"preset_source": source},
    )
    if len(presets) > 1:
        from menace.roi_tracker import ROITracker

        summary = ROITracker()
        for idx, preset in enumerate(presets):
            tracker = simulate_full_environment(preset)
            delta = sum(tracker.roi_history)
            sec_hist = tracker.metrics_history.get("security_score", [])
            sec_val = sec_hist[-1] if sec_hist else 0.0
            failing = getattr(tracker, "_last_test_failures", [])
            summary.update(
                0.0,
                delta,
                modules=[f"preset_{idx}"],
                metrics={
                    "security_score": sec_val,
                    "test_status": {name: False for name in failing},
                },
            )
        logger.info("sandbox presets complete", extra={"ranking": summary.rankings()})
        return

    preset = presets[0]
    builder = create_context_builder()
    sandbox_main(preset, args, builder)
    if _SandboxMetaLogger is not None:
        try:
            data_dir = Path(resolve_path(os.getenv("SANDBOX_DATA_DIR", "sandbox_data")))
            meta = _SandboxMetaLogger(data_dir / "sandbox_meta.log")
            for cyc, roi, succ, fail, dur, cov in meta.rankings():
                total = succ + fail
                rate = (succ / total * 100.0) if total else 0.0
                logger.info(
                    "cycle %d: ROI %.3f success_rate %.1f%% runtime %.2fs coverage %.1f%%",
                    cyc,
                    roi,
                    rate,
                    dur,
                    cov,
                    extra={
                        "cycle": cyc,
                        "roi": roi,
                        "success_rate": rate,
                        "duration": dur,
                        "coverage_percent": cov,
                    },
                )
        except Exception:
            logger.warning("failed to load sandbox meta log", exc_info=True)
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
            print(f"{name} {metric}={val:.3f}")
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


def foresight_trend(file: str, workflow_id: str) -> None:
    """Print slope, second derivative and volatility for ``workflow_id``.

    The ``file`` argument should point to a JSON list of metric dictionaries as
    recorded by :class:`ForesightTracker.record_cycle_metrics`.
    """

    tracker = ForesightTracker()
    try:
        with open(file) as fh:
            history = json.load(fh)
    except Exception:
        logger.exception("failed to load history %s", file)
        return

    for metrics in history:
        tracker.record_cycle_metrics(workflow_id, metrics)

    slope, second_derivative, volatility = tracker.get_trend_curve(workflow_id)
    print(
        json.dumps(
            {
                "slope": slope,
                "second_derivative": second_derivative,
                "volatility": volatility,
            },
            indent=2,
        )
    )


def foresight_stability(file: str, workflow_id: str) -> None:
    """Print whether the ROI trend for ``workflow_id`` is stable."""

    tracker = ForesightTracker()
    try:
        with open(file) as fh:
            history = json.load(fh)
    except Exception:
        logger.exception("failed to load history %s", file)
        return

    for metrics in history:
        tracker.record_cycle_metrics(workflow_id, metrics)

    stable = tracker.is_stable(workflow_id)
    print(json.dumps({"stable": stable}, indent=2))


def _capture_run(preset: dict[str, str], args: argparse.Namespace):
    """Run sandbox for ``preset`` and return the resulting tracker."""
    from sandbox_runner import _sandbox_main

    holder: dict[str, Any] = {}

    def wrapper(p: dict[str, str], a: argparse.Namespace, builder: Any) -> None:
        holder["tracker"] = _sandbox_main(p, a, builder)

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
    entropy_history: dict[str, list[float]] | None = None,
    entropy_threshold: float | None = None,
    entropy_consecutive: int | None = None,
) -> tuple[list[str], dict[str, float]]:
    """Return modules with ROI or entropy deltas that plateau within ``threshold``.

    The last ``consecutive`` ROI deltas are evaluated using an exponential moving
    average and variance. A module is flagged when the EMA magnitude is below the
    given ``threshold`` and the exponentially weighted standard deviation is less
    than ``std_threshold``. Additionally modules whose recent entropy delta ratios
    stay within ``threshold`` for ``consecutive`` entries are flagged.
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

    if entropy_history:
        e_thr = thr if entropy_threshold is None else float(entropy_threshold)
        e_consec = consecutive if entropy_consecutive is None else int(entropy_consecutive)
        for mod, vals in entropy_history.items():
            if mod in flagged or mod in flags or len(vals) < e_consec:
                continue
            window = vals[-e_consec:]
            if all(abs(v) <= e_thr for v in window):
                flags.append(mod)
                confidences[mod] = 1.0

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
            window_vals = series[start:i + 1]
            ma_series.append(sum(window_vals) / len(window_vals))
        try:  # pragma: no cover - optional dependency
            from statsmodels.tsa.stattools import adfuller

            adf_ps: list[float] = []
            size = min(len(ma_series), ma_window)
            if size >= 3:
                for j in range(len(ma_series) - size + 1):
                    p_val = adfuller(ma_series[j:j + size])[1]
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
                sub = series[i:i + size]
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
    settings = get_settings()
    if getattr(args, "dashboard_port", None):
        if MetricsDashboard is None:
            logger.warning(
                "MetricsDashboard unavailable; install Flask to enable the dashboard.",
                extra={"error": str(_METRICS_DASHBOARD_ERROR)},
            )
        else:
            history_dir = resolve_path(args.sandbox_data_dir or "sandbox_data")
            history_file = history_dir / "roi_history.json"
            dash = MetricsDashboard(str(history_file))
            Thread(
                target=dash.run, kwargs={"port": args.dashboard_port}, daemon=True
            ).start()

    module_history: dict[str, list[float]] = {}
    module_entropy_history: dict[str, list[float]] = {}
    flagged: set[str] = set()
    synergy_history = list(synergy_history or [])
    synergy_pred_history: list[dict[str, float]] = []
    roi_ma_history: list[float] = []
    synergy_ma_history = list(synergy_ma_history or [])
    roi_threshold = getattr(args, "roi_threshold", None)
    if roi_threshold is None:
        roi_threshold = settings.roi_threshold
    synergy_threshold = getattr(args, "synergy_threshold", None)
    if synergy_threshold is None:
        synergy_threshold = settings.synergy_threshold
    roi_k = getattr(args, "roi_threshold_k", None)
    if roi_k is None:
        roi_k = getattr(settings, "roi_threshold_k", 1.0)
    synergy_k = getattr(args, "synergy_threshold_k", None)
    if synergy_k is None:
        synergy_k = getattr(settings, "synergy_threshold_k", 1.0)
    roi_confidence = getattr(args, "roi_confidence", None)
    if roi_confidence is None:
        roi_confidence = settings.roi_confidence
    synergy_confidence = getattr(args, "synergy_confidence", None)
    if synergy_confidence is None:
        synergy_confidence = settings.synergy_confidence
    entropy_plateau_threshold = getattr(args, "entropy_plateau_threshold", None)
    if entropy_plateau_threshold is None:
        entropy_plateau_threshold = settings.entropy_plateau_threshold
    entropy_consecutive = getattr(args, "entropy_plateau_consecutive", None)
    if entropy_consecutive is None:
        entropy_consecutive = settings.entropy_plateau_consecutive
    entropy_threshold = getattr(args, "entropy_threshold", None)
    if entropy_threshold is None:
        entropy_threshold = settings.entropy_threshold
    entropy_ceiling_threshold = getattr(args, "entropy_ceiling_threshold", None)
    if entropy_ceiling_threshold is None:
        entropy_ceiling_threshold = settings.entropy_ceiling_threshold
    if entropy_ceiling_threshold is not None:
        os.environ["ENTROPY_CEILING_THRESHOLD"] = str(entropy_ceiling_threshold)
    entropy_ceiling_consecutive = getattr(args, "entropy_ceiling_consecutive", None)
    if entropy_ceiling_consecutive is None:
        entropy_ceiling_consecutive = settings.entropy_ceiling_consecutive
    if entropy_ceiling_consecutive is not None:
        os.environ["ENTROPY_CEILING_CONSECUTIVE"] = str(entropy_ceiling_consecutive)
    synergy_threshold_window = getattr(args, "synergy_threshold_window", None)
    if synergy_threshold_window is None:
        synergy_threshold_window = settings.synergy_threshold_window
    synergy_threshold_weight = getattr(args, "synergy_threshold_weight", None)
    if synergy_threshold_weight is None:
        synergy_threshold_weight = settings.synergy_threshold_weight
    synergy_ma_window = getattr(args, "synergy_ma_window", None)
    if synergy_ma_window is None:
        synergy_ma_window = settings.synergy_ma_window
    synergy_stationarity_confidence = getattr(
        args, "synergy_stationarity_confidence", None
    )
    if synergy_stationarity_confidence is None:
        synergy_stationarity_confidence = settings.synergy_stationarity_confidence
    synergy_std_threshold = getattr(args, "synergy_std_threshold", None)
    if synergy_std_threshold is None:
        synergy_std_threshold = settings.synergy_std_threshold
    synergy_variance_confidence = getattr(args, "synergy_variance_confidence", None)
    if synergy_variance_confidence is None:
        synergy_variance_confidence = settings.synergy_variance_confidence
    last_tracker = None
    iteration = 0
    roi_cycles = getattr(args, "roi_cycles", settings.roi_cycles or 3)
    synergy_cycles = getattr(args, "synergy_cycles", settings.synergy_cycles or 3)
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
            if entropy_threshold is not None:
                os.environ["SANDBOX_ENTROPY_THRESHOLD"] = str(entropy_threshold)
            run_args = argparse.Namespace(
                sandbox_data_dir=args.sandbox_data_dir,
                workflow_db=str(resolve_path("workflows.db")),
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
            for mod, vals in getattr(tracker, "module_entropy_deltas", {}).items():
                module_entropy_history.setdefault(mod, []).extend(vals)
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
                win = min(len(last_tracker.roi_history), roi_cycles)
                roi_threshold = _adaptive_threshold(
                    last_tracker.roi_history,
                    win,
                )
            elif roi_threshold is None:
                if hasattr(last_tracker, "get") and hasattr(last_tracker, "std"):
                    try:
                        roi_threshold = (
                            last_tracker.get("roi_delta")
                            + roi_k * last_tracker.std("roi_delta")
                        )
                    except Exception:
                        roi_threshold = last_tracker.diminishing()
                else:
                    roi_threshold = last_tracker.diminishing()
            new_flags, _ = _diminishing_modules(
                module_history,
                flagged,
                roi_threshold,
                consecutive=roi_cycles,
                confidence=roi_confidence or 0.95,
                entropy_history=module_entropy_history,
                entropy_threshold=entropy_plateau_threshold,
                entropy_consecutive=entropy_consecutive,
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
            if hasattr(last_tracker, "get") and hasattr(last_tracker, "std"):
                try:
                    base = last_tracker.get("synergy_roi")
                    dev = last_tracker.std("synergy_roi")
                    synergy_threshold = base + synergy_k * dev
                except Exception:
                    synergy_threshold = last_tracker.diminishing()
            else:
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
        all_mods = set(module_history) | set(module_entropy_history)
        if all_mods and all_mods <= flagged and converged:
            logger.info(
                "synergy convergence reached",
                extra={"iteration": iteration, "ema": ema_val},
            )
            break

    if last_tracker:
        logger.info("=== Final Module Rankings ===", extra={"iteration": iteration})
        for mod, raroi, roi in last_tracker.rankings():
            logger.info(
                "%s: RAROI %.3f ROI %.3f",
                mod,
                raroi,
                roi,
                extra={
                    "iteration": iteration,
                    "module_name": mod,
                    "raroi": raroi,
                    "roi": roi,
                },
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

    def _gen_presets(n=None):
        return presets

    generate_presets = _gen_presets  # type: ignore
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

    system = platform.system().lower()

    if system == "windows":
        xml = resolve_path("systemd/windows_sandbox_purge.xml")
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
        service = resolve_path("systemd/sandbox_autopurge.service")
        timer = resolve_path("systemd/sandbox_autopurge.timer")
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
    print("[QFE:sandbox] _cli_main entered", flush=True)
    print("[QFE:sandbox] retrieving settings", flush=True)
    set_correlation_id(str(uuid.uuid4()))
    settings = get_settings()
    print("[QFE:sandbox] settings loaded", flush=True)
    parser = argparse.ArgumentParser(description="Run Menace sandbox")
    parser.add_argument(
        "--workflow-sim",
        action="store_true",
        help="simulate workflows instead of repo sections",
    )
    parser.add_argument(
        "--workflow-db",
        default=str(resolve_path("workflows.db")),
        help="path to workflow database",
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
        default=settings.sandbox_module_algo,
        help="module clustering algorithm",
    )
    parser.add_argument(
        "--module-threshold",
        type=float,
        default=settings.sandbox_module_threshold,
        help="semantic similarity threshold",
    )
    parser.add_argument(
        "--module-semantic",
        action="store_true",
        default=settings.sandbox_semantic_modules,
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
        "--print-scenario-summary",
        action="store_true",
        help="display scenario summary after runs",
    )
    parser.add_argument(
        "--run-scenarios",
        type=int,
        metavar="WORKFLOW_ID",
        help="execute predefined scenario simulations for the given workflow and exit",
    )
    parser.add_argument(
        "--simulate-temporal-trajectory",
        type=int,
        metavar="WORKFLOW_ID",
        help="simulate temporal trajectory for the given workflow and exit",
    )
    parser.add_argument(
        "--alignment-warnings",
        action="store_true",
        help="display recent alignment warnings and exit",
    )
    parser.add_argument(
        "--fail-on-missing-scenarios",
        action="store_true",
        help=(
            "treat missing canonical scenarios as errors "
            "(or set SANDBOX_FAIL_ON_MISSING_SCENARIOS=1)"
        ),
    )
    parser.add_argument(
        "--max-recursion-depth",
        type=int,
        default=settings.sandbox_max_recursion_depth,
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
        help=(
            "skip modules listed in "
            f"{path_for_prompt('sandbox_data')}/orphan_modules.json"
        ),
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
        "--entropy-threshold",
        "--entropy-plateau-threshold",
        dest="entropy_plateau_threshold",
        type=float,
        help="threshold for entropy delta plateau detection",
    )
    parser.add_argument(
        "--consecutive",
        "--entropy-plateau-consecutive",
        dest="entropy_plateau_consecutive",
        type=int,
        help="entropy delta samples below threshold before module convergence",
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
        default=settings.sandbox_log_level or settings.log_level,
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
        default=str(resolve_path("sandbox_data") / "roi_history.json"),
        help="path to roi_history.json",
    )
    p_metrics.add_argument(
        "--window",
        type=int,
        default=5,
        help="EMA window for moving average",
    )
    p_metrics.add_argument("--plot", action="store_true", help="show matplotlib plot")

    p_trend = sub.add_parser(
        "foresight-trend", help="show ROI trend metrics from history"
    )
    p_trend.add_argument("--file", required=True, help="path to metrics history")
    p_trend.add_argument("--workflow-id", default="wf", help="workflow identifier")

    p_stable = sub.add_parser(
        "foresight-stable", help="check workflow stability from history"
    )
    p_stable.add_argument("--file", required=True, help="path to metrics history")
    p_stable.add_argument(
        "--workflow-id", default="wf", help="workflow identifier"
    )

    sub.add_parser(
        "relevancy-report",
        help="print modules flagged by relevancy radar",
    )

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
        conflict_handler="resolve",
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
        "--roi-threshold-k",
        type=float,
        help="std dev multiplier when estimating ROI threshold",
    )
    p_autorun.add_argument(
        "--roi-confidence",
        type=float,
        help="confidence level for ROI convergence",
    )
    p_autorun.add_argument(
        "--entropy-threshold",
        "--entropy-plateau-threshold",
        dest="entropy_plateau_threshold",
        type=float,
        help="threshold for entropy delta plateau detection",
    )
    p_autorun.add_argument(
        "--entropy-consecutive",
        "--entropy-plateau-consecutive",
        "--consecutive",
        dest="entropy_plateau_consecutive",
        type=int,
        help="entropy delta samples below threshold before module convergence",
    )
    p_autorun.add_argument(
        "--entropy-threshold",
        type=float,
        help="ROI gain per entropy delta threshold",
    )
    p_autorun.add_argument(
        "--entropy-ceiling-threshold",
        type=float,
        help="ROI/entropy ratio ceiling before module retirement",
    )
    p_autorun.add_argument(
        "--entropy-ceiling-consecutive",
        type=int,
        help="cycles below ceiling threshold before retirement",
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
        "--synergy-threshold-k",
        type=float,
        help="std dev multiplier when estimating synergy threshold",
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
        "--roi-threshold-k",
        type=float,
        help="std dev multiplier when estimating ROI threshold",
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
        "--synergy-threshold-k",
        type=float,
        help="std dev multiplier when estimating synergy threshold",
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

    env_settings = get_settings()
    auto_include_isolated = bool(
        getattr(env_settings, "auto_include_isolated", True)
        or getattr(args, "auto_include_isolated", False)
    )
    recursive_orphans = getattr(env_settings, "recursive_orphan_scan", True)
    if args.recursive_orphans is not None:
        recursive_orphans = args.recursive_orphans
    recursive_isolated = getattr(env_settings, "recursive_isolated", True)
    if args.recursive_isolated is not None:
        recursive_isolated = args.recursive_isolated
    if auto_include_isolated:
        recursive_isolated = True

    fail_on_missing = bool(
        getattr(env_settings, "fail_on_missing_scenarios", False)
        or getattr(args, "fail_on_missing_scenarios", False)
    )

    args.auto_include_isolated = auto_include_isolated
    args.recursive_orphans = recursive_orphans
    args.recursive_isolated = recursive_isolated
    args.fail_on_missing_scenarios = fail_on_missing

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
    os.environ["SANDBOX_FAIL_ON_MISSING_SCENARIOS"] = "1" if fail_on_missing else "0"

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
    if getattr(args, "entropy_plateau_threshold", None) is not None:
        os.environ["ENTROPY_PLATEAU_THRESHOLD"] = str(
            args.entropy_plateau_threshold
        )
    if getattr(args, "entropy_plateau_consecutive", None) is not None:
        os.environ["ENTROPY_PLATEAU_CONSECUTIVE"] = str(
            args.entropy_plateau_consecutive
        )

    if getattr(args, "alignment_warnings", False):
        from menace import violation_logger as vl
        print(json.dumps(vl.load_recent_alignment_warnings(), indent=2))
        return

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

    if getattr(args, "cmd", None) == "foresight-trend":
        foresight_trend(args.file, args.workflow_id)
        return

    if getattr(args, "cmd", None) == "foresight-stable":
        foresight_stability(args.file, args.workflow_id)
        return

    if getattr(args, "cmd", None) == "relevancy-report":
        from relevancy_radar import flagged_modules
        from metrics_exporter import update_relevancy_metrics

        flags = flagged_modules()
        update_relevancy_metrics(flags)
        if not flags:
            print("No modules flagged.")
        else:
            for mod, flag in sorted(flags.items()):
                print(f"{mod}: {flag}")
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

    if args.simulate_temporal_trajectory is not None:
        from sandbox_runner.environment import simulate_temporal_trajectory
        from task_handoff_bot import WorkflowDB

        wf_db = WorkflowDB(Path(args.workflow_db))
        row = wf_db.conn.execute(
            "SELECT workflow, task_sequence FROM workflows WHERE id=?",
            (args.simulate_temporal_trajectory,),
        ).fetchone()
        if not row:
            print(f"workflow {args.simulate_temporal_trajectory} not found")
            return
        steps_raw = row["workflow"] if isinstance(row, Mapping) else row[0]
        if not steps_raw:
            steps_raw = row["task_sequence"] if isinstance(row, Mapping) else row[1]
        workflow_steps = [s for s in str(steps_raw).split(",") if s]

        ft = ForesightTracker()
        simulate_temporal_trajectory(
            str(args.simulate_temporal_trajectory),
            workflow_steps,
            foresight_tracker=ft,
        )
        path = (
            resolve_path("sandbox_data")
            / f"temporal_trajectory_{args.simulate_temporal_trajectory}.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        data = ft.to_dict()
        path.write_text(json.dumps(data, indent=2))
        print(json.dumps(data, indent=2))
        return

    if args.run_scenarios is not None:
        from sandbox_runner.environment import run_scenarios
        from task_handoff_bot import WorkflowDB

        wf_db = WorkflowDB(Path(args.workflow_db))
        row = wf_db.conn.execute(
            "SELECT * FROM workflows WHERE id=?",
            (args.run_scenarios,),
        ).fetchone()
        if not row:
            print(f"workflow {args.run_scenarios} not found")
            return
        wf = wf_db._row_to_record(row)
        _, _, summary = run_scenarios(wf)
        worst = summary.get("worst_scenario")
        for scen, info in sorted(summary.get("scenarios", {}).items()):
            marker = " <== WORST" if scen == worst else ""
            print(f"{scen}: {info['roi_delta']:+.3f}{marker}")
        if worst:
            print(f"worst_scenario: {worst}")
        return

    if args.workflow_sim:
        from sandbox_runner.environment import (
            run_workflow_simulations,
            load_scenario_summary,
        )
        builder = create_context_builder()
        run_workflow_simulations(
            args.workflow_db,
            dynamic_workflows=args.dynamic_workflows,
            module_algorithm=args.module_algorithm or "greedy",
            module_threshold=float(args.module_threshold)
            if args.module_threshold is not None
            else 0.1,
            module_semantic=args.module_semantic,
            context_builder=builder,
        )
        if getattr(args, "print_scenario_summary", False):
            print(json.dumps(load_scenario_summary(), indent=2))
    else:
        _run_sandbox(args)
        if getattr(args, "print_scenario_summary", False):
            from sandbox_runner.environment import load_scenario_summary

            print(json.dumps(load_scenario_summary(), indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI usage
    setup_logging()
    main()
