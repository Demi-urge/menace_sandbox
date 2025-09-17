"""Utilities for self-coding ROI/error thresholds.

The configuration file ``config/self_coding_thresholds.yaml`` stores
thresholds keyed by bot name.  Each entry may define ``roi_drop``
(allowable decrease in ROI), ``error_increase`` (maximum allowed
increase in errors), ``test_failure_increase`` (allowed growth in
failing tests), ``test_command`` (custom test runner) and
``workflow_tests`` (pytest selectors or scripts covering key flows).
Forecast configuration is also stored allowing each bot to select a
forecasting model, confidence interval and optional parameters via
``model``, ``confidence`` and ``model_params`` fields.

Long running services cache threshold values after the first lookup.
When the YAML file is edited at runtime these values can be refreshed by
calling :meth:`menace.data_bot.DataBot.reload_thresholds` for the
affected bot.  This function simply wraps :func:`get_thresholds` and
updates the in-memory cache, allowing new limits to take effect without a
restart.

Example configuration snippet::

    default:
      roi_drop: -0.1
      error_increase: 1.0
      test_failure_increase: 0.0
      model: exponential
      confidence: 0.95
      model_params: {alpha: 0.3}
      test_command: ["pytest", "-q"]
      workflow_tests: ["tests/workflows/test_default.py::test_smoke"]
    bots:
      example-bot:
        roi_drop: -0.2
        error_increase: 2.0
        test_failure_increase: 0.1
        model: arima
        confidence: 0.9
        model_params: {order: [1, 1, 1]}
        test_command: ["pytest", "tests/example", "-q"]
        workflow_tests: ["python", "scripts/run_example_workflow.py"]

You can modify the file directly or use :func:`update_thresholds`::

    from menace.self_coding_thresholds import update_thresholds
    update_thresholds("example-bot", roi_drop=-0.25)

After editing the file, apply the changes at runtime::

    from menace.data_bot import DataBot
    bot = DataBot()
    bot.reload_thresholds("example-bot")
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import os
import shlex
import yaml

from .sandbox_settings import SandboxSettings, normalize_workflow_tests
from .dynamic_path_router import resolve_path
from .roi_thresholds import ROIThresholds

try:  # pragma: no cover - optional dependency
    from .trend_predictor import TrendPredictor, TrendPrediction
except Exception:  # pragma: no cover - predictor is optional
    TrendPredictor = TrendPrediction = None  # type: ignore


@dataclass(frozen=True)
class SelfCodingThresholds:
    """Container for self-coding trigger thresholds."""

    roi_drop: float
    error_increase: float
    test_failure_increase: float = 0.0
    patch_success_drop: float = -0.2
    roi_weight: float = 0.5
    error_weight: float = 0.3
    test_failure_weight: float = 0.2
    patch_success_weight: float = 0.1
    auto_recalibrate: bool = True
    test_command: list[str] | None = None
    workflow_tests: list[str] | None = None
    model: str = "exponential"
    confidence: float = 0.95
    model_params: dict | None = None


_CONFIG_PATH = resolve_path("config/self_coding_thresholds.yaml")


def _load_config(path: Path | None = None) -> Dict[str, dict]:
    try:
        return yaml.safe_load((path or _CONFIG_PATH).read_text()) or {}
    except Exception:
        return {}


def get_thresholds(
    bot: str | None = None,
    settings: SandboxSettings | None = None,
    *,
    path: Path | None = None,
) -> SelfCodingThresholds:
    """Return thresholds for ``bot``.

    Parameters
    ----------
    bot:
        Bot name to lookup.  When omitted the default thresholds are
        returned.
    settings:
        Optional ``SandboxSettings`` providing baseline values when the
        configuration file is missing.
    path:
        Optional override for the configuration path.
    """

    s = settings or SandboxSettings()
    roi_drop = getattr(s, "self_coding_roi_drop", -0.1)
    err_inc = getattr(s, "self_coding_error_increase", 1.0)
    fail_inc = getattr(s, "self_coding_test_failure_increase", 0.0)
    patch_drop = getattr(s, "self_coding_patch_success_drop", -0.2)
    roi_weight = getattr(s, "self_coding_roi_weight", 0.5)
    err_weight = getattr(s, "self_coding_error_weight", 0.3)
    fail_weight = getattr(s, "self_coding_test_failure_weight", 0.2)
    patch_weight = getattr(s, "self_coding_patch_success_weight", 0.1)
    auto_recal = getattr(s, "self_coding_auto_recalibrate", True)
    cmd: list[str] | None
    env_cmd = os.getenv("SELF_CODING_TEST_COMMAND")
    if env_cmd:
        cmd = shlex.split(env_cmd)
    else:
        cmd_val = getattr(s, "self_coding_test_command", None)
        if isinstance(cmd_val, str):
            cmd = shlex.split(cmd_val)
        else:
            cmd = cmd_val
    workflow_tests: list[str] = []
    workflow_cfg_set = False

    def _override_from_value(value: Any, *, explicit: bool) -> None:
        nonlocal workflow_tests, workflow_cfg_set
        normalized = normalize_workflow_tests(value)
        if normalized or explicit:
            workflow_tests = list(normalized)
            if explicit:
                workflow_cfg_set = True
            else:
                workflow_cfg_set = workflow_cfg_set or bool(normalized)

    def _override_from_bt(bt: Any) -> None:
        if bt is None:
            return
        fields_set = getattr(bt, "__fields_set__", set())
        explicit = "workflow_tests" in fields_set
        _override_from_value(getattr(bt, "workflow_tests", None), explicit=explicit)

    thresholds_cfg = getattr(s, "bot_thresholds", {})
    if isinstance(thresholds_cfg, dict):
        _override_from_bt(thresholds_cfg.get("default"))
        if bot:
            _override_from_bt(thresholds_cfg.get(bot))

    data = _load_config(path)
    default = data.get("default", {})
    bots = data.get("bots", {})
    roi_drop = float(default.get("roi_drop", roi_drop))
    err_inc = float(default.get("error_increase", err_inc))
    fail_inc = float(default.get("test_failure_increase", fail_inc))
    patch_drop = float(default.get("patch_success_drop", patch_drop))
    roi_weight = float(default.get("roi_weight", roi_weight))
    err_weight = float(default.get("error_weight", err_weight))
    fail_weight = float(default.get("test_failure_weight", fail_weight))
    patch_weight = float(default.get("patch_success_weight", patch_weight))
    auto_recal = bool(default.get("auto_recalibrate", auto_recal))
    model = default.get("model", "exponential")
    conf = float(default.get("confidence", 0.95))
    params = default.get("model_params", {})
    cmd_cfg = default.get("test_command", cmd)
    if "workflow_tests" in default:
        _override_from_value(default.get("workflow_tests"), explicit=True)
    if bot and bot in bots:
        cfg = bots[bot] or {}
        roi_drop = float(cfg.get("roi_drop", roi_drop))
        err_inc = float(cfg.get("error_increase", err_inc))
        fail_inc = float(cfg.get("test_failure_increase", fail_inc))
        patch_drop = float(cfg.get("patch_success_drop", patch_drop))
        roi_weight = float(cfg.get("roi_weight", roi_weight))
        err_weight = float(cfg.get("error_weight", err_weight))
        fail_weight = float(cfg.get("test_failure_weight", fail_weight))
        patch_weight = float(cfg.get("patch_success_weight", patch_weight))
        auto_recal = bool(cfg.get("auto_recalibrate", auto_recal))
        model = cfg.get("model", model)
        conf = float(cfg.get("confidence", conf))
        params = cfg.get("model_params", params)
        cmd_cfg = cfg.get("test_command", cmd_cfg)
        if "workflow_tests" in cfg:
            _override_from_value(cfg.get("workflow_tests"), explicit=True)
    if isinstance(cmd_cfg, str):
        cmd_cfg = shlex.split(cmd_cfg)
    elif cmd_cfg is not None:
        cmd_cfg = list(cmd_cfg)
    if not isinstance(params, dict):
        params = {}
    final_workflow_tests = workflow_tests if (workflow_cfg_set or workflow_tests) else None
    return SelfCodingThresholds(
        roi_drop=roi_drop,
        error_increase=err_inc,
        test_failure_increase=fail_inc,
        patch_success_drop=patch_drop,
        roi_weight=roi_weight,
        error_weight=err_weight,
        test_failure_weight=fail_weight,
        patch_success_weight=patch_weight,
        auto_recalibrate=auto_recal,
        test_command=cmd_cfg,
        workflow_tests=final_workflow_tests,
        model=model,
        confidence=conf,
        model_params=params,
    )


def update_thresholds(
    bot: str,
    *,
    roi_drop: float | None = None,
    error_increase: float | None = None,
    test_failure_increase: float | None = None,
    patch_success_drop: float | None = None,
    roi_weight: float | None = None,
    error_weight: float | None = None,
    test_failure_weight: float | None = None,
    patch_success_weight: float | None = None,
    auto_recalibrate: bool | None = None,
    test_command: list[str] | None = None,
    workflow_tests: list[str] | None = None,
    forecast_model: str | None = None,
    confidence: float | None = None,
    forecast_params: dict | None = None,
    path: Path | None = None,
) -> None:
    """Persist new thresholds for ``bot`` to the configuration file."""

    cfg_path = path or _CONFIG_PATH
    data = _load_config(cfg_path)
    bots = data.setdefault("bots", {})
    cfg = bots.setdefault(bot, {})
    if roi_drop is not None:
        cfg["roi_drop"] = float(roi_drop)
    if error_increase is not None:
        cfg["error_increase"] = float(error_increase)
    if test_failure_increase is not None:
        cfg["test_failure_increase"] = float(test_failure_increase)
    if patch_success_drop is not None:
        cfg["patch_success_drop"] = float(patch_success_drop)
    if roi_weight is not None:
        cfg["roi_weight"] = float(roi_weight)
    if error_weight is not None:
        cfg["error_weight"] = float(error_weight)
    if test_failure_weight is not None:
        cfg["test_failure_weight"] = float(test_failure_weight)
    if patch_success_weight is not None:
        cfg["patch_success_weight"] = float(patch_success_weight)
    if auto_recalibrate is not None:
        cfg["auto_recalibrate"] = bool(auto_recalibrate)
    if test_command is not None:
        cfg["test_command"] = list(test_command)
    if workflow_tests is not None:
        cfg["workflow_tests"] = list(workflow_tests)
    if forecast_model is not None:
        cfg["model"] = forecast_model
    if confidence is not None:
        cfg["confidence"] = float(confidence)
    if forecast_params is not None:
        cfg["model_params"] = dict(forecast_params)
    try:
        cfg_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    except Exception:
        # best effort – calling code may log failures
        pass


def adaptive_thresholds(
    bot: str,
    *,
    roi_baseline: float,
    error_baseline: float,
    failure_baseline: float,
    prediction: "TrendPrediction | None" = None,
    predictor: "TrendPredictor | None" = None,
) -> ROIThresholds:
    """Derive and persist adaptive thresholds for ``bot``.

    ``roi_baseline`` and friends represent rolling averages of recent
    metrics.  ``prediction`` may supply a forecast from
    :class:`~menace.trend_predictor.TrendPredictor`; when omitted a predictor
    instance is used to generate one.  The returned
    :class:`~menace.roi_thresholds.ROIThresholds` reflects updated limits that
    have also been written back to ``self_coding_thresholds.yaml`` so future
    lookups share the same view.
    """

    if prediction is None and TrendPredictor is not None:
        pred = predictor or TrendPredictor()
        try:
            pred.train()
            prediction = pred.predict_future_metrics()
        except Exception:  # pragma: no cover - prediction errors
            prediction = None

    pred_roi = prediction.roi if prediction else roi_baseline
    pred_err = prediction.errors if prediction else error_baseline
    pred_fail = failure_baseline  # tests_failed forecasts are currently
    # unavailable so fall back to the rolling baseline.

    current = get_thresholds(bot)
    roi_thresh = min(current.roi_drop, pred_roi - roi_baseline)
    err_thresh = max(current.error_increase, pred_err - error_baseline)
    fail_thresh = max(current.test_failure_increase, pred_fail - failure_baseline)

    update_thresholds(
        bot,
        roi_drop=roi_thresh,
        error_increase=err_thresh,
        test_failure_increase=fail_thresh,
        patch_success_drop=current.patch_success_drop,
    )

    return ROIThresholds(
        roi_drop=roi_thresh,
        error_threshold=err_thresh,
        test_failure_threshold=fail_thresh,
        patch_success_drop=current.patch_success_drop,
    )


__all__ = [
    "SelfCodingThresholds",
    "get_thresholds",
    "update_thresholds",
    "adaptive_thresholds",
]
