from __future__ import annotations

"""Lightweight sandbox settings implementation used when pydantic is unavailable."""

from dataclasses import asdict, dataclass, field, make_dataclass
from functools import lru_cache
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence, Union, get_args, get_origin

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - optional
    yaml = None  # type: ignore
    _MISSING_YAML = exc
else:
    _MISSING_YAML = None

try:  # pragma: no cover - prefer package-relative import
    from .dynamic_path_router import resolve_path
except Exception:  # pragma: no cover - support flat execution layout
    from dynamic_path_router import resolve_path  # type: ignore

try:  # pragma: no cover - prefer colocated helper when packaged
    from .stack_dataset_defaults import (
        STACK_LANGUAGE_ALLOWLIST,
        normalise_stack_languages,
    )
except Exception:  # pragma: no cover - allow direct module import when flat
    from stack_dataset_defaults import (  # type: ignore
        STACK_LANGUAGE_ALLOWLIST,
        normalise_stack_languages,
    )

PYDANTIC_V2 = False
USING_SANDBOX_SETTINGS_FALLBACK = True

_DEFAULTS_PATH = Path(__file__).with_name("sandbox_settings_defaults.json")
_ENV_MAP_PATH = Path(__file__).with_name("sandbox_settings_env_map.json")
_MODEL_DEFAULTS_PATH = Path(__file__).with_name("sandbox_settings_model_defaults.json")

with _DEFAULTS_PATH.open("r", encoding="utf-8") as fh:
    _SANDBOX_DEFAULTS: dict[str, Any] = json.load(fh)
with _ENV_MAP_PATH.open("r", encoding="utf-8") as fh:
    _SANDBOX_ENV_MAP: dict[str, list[str]] = json.load(fh)
with _MODEL_DEFAULTS_PATH.open("r", encoding="utf-8") as fh:
    _MODEL_DEFAULTS: dict[str, dict[str, Any]] = json.load(fh)

DEFAULT_SEVERITY_SCORE_MAP: dict[str, float] = dict(
    _SANDBOX_DEFAULTS.get("severity_score_map", {})
)


def _coerce_optional_bool(value: Any) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        lowered = candidate.lower()
        if lowered in {"none", "null"}:
            return None
        if lowered in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "f", "no", "n", "off"}:
            return False
        return False
    return bool(value)


def _validate_stack_languages(languages: list[str]) -> list[str]:
    unknown = [lang for lang in languages if lang not in STACK_LANGUAGE_ALLOWLIST]
    if unknown:
        allowed = ", ".join(sorted(STACK_LANGUAGE_ALLOWLIST))
        raise ValueError(
            "Unsupported Stack dataset languages: "
            f"{', '.join(sorted(set(unknown)))}. Allowed values are: {allowed}"
        )
    return languages


def normalize_workflow_tests(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return []
        try:
            parsed = json.loads(candidate)
        except Exception:
            parts = [item.strip() for item in candidate.split(",") if item.strip()]
            if len(parts) > 1:
                return parts
            return [candidate]
        if isinstance(parsed, str):
            parsed = parsed.strip()
            return [parsed] if parsed else []
        if isinstance(parsed, (list, tuple, set)):
            return [
                str(item).strip()
                for item in parsed
                if str(item).strip()
            ]
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    coerced = str(value).strip()
    return [coerced] if coerced else []


def _deep_copy(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _deep_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deep_copy(item) for item in value]
    if isinstance(value, set):
        return {_deep_copy(item) for item in value}
    if isinstance(value, tuple):
        return tuple(_deep_copy(item) for item in value)
    return value


def _coerce_sequence(value: Sequence[Any], target_type: Any) -> list[Any]:
    inner_type = Any
    args = get_args(target_type)
    if args:
        inner_type = args[0]
    return [_coerce_value(item, inner_type) for item in value]


def _coerce_mapping(value: Mapping[str, Any], target_type: Any) -> dict[str, Any]:
    key_type, val_type = Any, Any
    args = get_args(target_type)
    if len(args) == 2:
        key_type, val_type = args
    coerced: dict[str, Any] = {}
    for key, val in value.items():
        coerced[str(_coerce_value(key, key_type))] = _coerce_value(val, val_type)
    return coerced


def _coerce_value(value: Any, target_type: Any) -> Any:
    origin = get_origin(target_type)
    if origin is None:
        if target_type in {bool, None.__class__}:
            result = _coerce_optional_bool(value)
            return result if result is not None else False
        if target_type is int:
            return int(value)
        if target_type is float:
            return float(value)
        if target_type is str:
            return str(value)
        return value
    if origin in {list, tuple, set, Sequence}:
        sequence = value
        if isinstance(value, str):
            try:
                sequence = json.loads(value)
            except Exception:
                sequence = [item.strip() for item in value.split(",") if item.strip()]
        if not isinstance(sequence, Sequence):
            return [_coerce_value(sequence, Any)]
        return _coerce_sequence(sequence, target_type)
    if origin in {dict, Mapping}:
        mapping = value
        if isinstance(value, str):
            try:
                mapping = json.loads(value)
            except Exception:
                return {}
        if not isinstance(mapping, Mapping):
            return {}
        return _coerce_mapping(mapping, target_type)
    if origin is Union:
        for arg in get_args(target_type):
            if arg is type(None):
                continue
            try:
                return _coerce_value(value, arg)
            except Exception:
                continue
        return value
    return value


class _BaseModel:
    """Dataclass-compatible base providing ``model_dump`` helpers."""

    def model_dump(self, mode: str = "python") -> dict[str, Any]:  # pragma: no cover - simple
        return asdict(self)

    def dict(self) -> dict[str, Any]:  # pragma: no cover - alias
        return self.model_dump()

def _make_model(name: str) -> type[_BaseModel]:
    defaults = _MODEL_DEFAULTS[name]
    fields_spec = []
    for key, value in defaults.items():
        if isinstance(value, list):
            default = field(default_factory=lambda v=value: list(v))
        elif isinstance(value, dict):
            default = field(default_factory=lambda v=value: dict(v))
        else:
            default = field(default=value)
        fields_spec.append((key, Any, default))
    cls = make_dataclass(name, fields_spec, bases=(_BaseModel,))
    cls.__module__ = __name__
    return cls


AlignmentRules = _make_model("AlignmentRules")
ROISettings = _make_model("ROISettings")
BotThresholds = _make_model("BotThresholds")
SynergySettings = _make_model("SynergySettings")
AlignmentSettings = _make_model("AlignmentSettings")
AutoMergeSettings = _make_model("AutoMergeSettings")
ActorCriticSettings = _make_model("ActorCriticSettings")
PolicySettings = _make_model("PolicySettings")


class SandboxSettings:
    """Fallback configuration loader relying on pre-generated defaults."""

    __slots__ = ("_data",)

    def __init__(self, **overrides: Any) -> None:
        data = _deep_copy(_SANDBOX_DEFAULTS)
        data.update(self._load_env_overrides())
        data.update(overrides)
        if "stack_languages" in data:
            try:
                languages = normalise_stack_languages(data["stack_languages"])
                data["stack_languages"] = _validate_stack_languages(languages)
            except Exception:
                data["stack_languages"] = []
        if "stack_streaming" in data:
            data["stack_streaming"] = _coerce_optional_bool(data["stack_streaming"])
        self._data = data
        self._initialise_grouped_models()

    def _load_env_overrides(self) -> dict[str, Any]:
        overrides: dict[str, Any] = {}
        for key, env_names in _SANDBOX_ENV_MAP.items():
            candidates = env_names if isinstance(env_names, list) else [env_names]
            for env_name in candidates:
                if not env_name:
                    continue
                raw = os.getenv(env_name)
                if raw is None:
                    continue
                default = _SANDBOX_DEFAULTS.get(key)
                try:
                    coerced = _coerce_value(raw, type(default) if default is not None else Any)
                except Exception:
                    continue
                overrides[key] = coerced
                break
        return overrides

    def _initialise_grouped_models(self) -> None:
        d = self._data
        d["roi"] = ROISettings(
            threshold=d.get("roi_threshold"),
            confidence=d.get("roi_confidence"),
            ema_alpha=d.get("roi_ema_alpha"),
            compounding_weight=d.get("roi_compounding_weight"),
            min_integration_roi=d.get("min_integration_roi"),
            entropy_window=d.get("entropy_window"),
            entropy_weight=d.get("entropy_weight"),
            entropy_threshold=d.get("entropy_threshold"),
            entropy_plateau_threshold=d.get("entropy_plateau_threshold"),
            entropy_plateau_consecutive=d.get("entropy_plateau_consecutive"),
            entropy_ceiling_threshold=d.get("entropy_ceiling_threshold"),
            entropy_ceiling_consecutive=d.get("entropy_ceiling_consecutive"),
            baseline_window=d.get("roi_baseline_window", d.get("baseline_window")),
            deviation_tolerance=d.get("roi_deviation_tolerance", d.get("deviation_tolerance")),
            stagnation_threshold=d.get("roi_stagnation_threshold", d.get("stagnation_threshold")),
            momentum_window=d.get("roi_momentum_window", d.get("momentum_window")),
            stagnation_cycles=d.get("roi_stagnation_cycles", d.get("stagnation_cycles")),
            momentum_dev_multiplier=d.get("roi_momentum_dev_multiplier", d.get("momentum_dev_multiplier")),
            roi_stagnation_dev_multiplier=d.get("roi_stagnation_dev_multiplier"),
        )
        d["synergy"] = SynergySettings(
            threshold=d.get("synergy_threshold"),
            confidence=d.get("synergy_confidence"),
            threshold_window=d.get("synergy_threshold_window"),
            threshold_weight=d.get("synergy_threshold_weight"),
            ma_window=d.get("synergy_ma_window"),
            stationarity_confidence=d.get("synergy_stationarity_confidence"),
            std_threshold=d.get("synergy_std_threshold"),
            variance_confidence=d.get("synergy_variance_confidence"),
            weight_roi=d.get("synergy_weight_roi"),
            weight_efficiency=d.get("synergy_weight_efficiency"),
            weight_resilience=d.get("synergy_weight_resilience"),
            weight_antifragility=d.get("synergy_weight_antifragility"),
            weight_reliability=d.get("synergy_weight_reliability"),
            weight_maintainability=d.get("synergy_weight_maintainability"),
            weight_throughput=d.get("synergy_weight_throughput"),
            weights_lr=d.get("synergy_weights_lr"),
            train_interval=d.get("synergy_train_interval"),
            replay_size=d.get("synergy_replay_size"),
            batch_size=d.get("synergy_batch_size"),
            gamma=d.get("synergy_gamma"),
            noise=d.get("synergy_noise"),
            hidden_size=d.get("synergy_hidden_size"),
            layers=d.get("synergy_layers"),
            optimizer=d.get("synergy_optimizer"),
            checkpoint_interval=d.get("synergy_checkpoint_interval"),
            strategy=d.get("synergy_strategy"),
            target_sync=d.get("synergy_target_sync"),
            python_fallback=d.get("synergy_python_fallback"),
            python_max_replay=d.get("synergy_python_max_replay"),
        )
        d["alignment"] = AlignmentSettings(
            rules=AlignmentRules(),
            enable_flagger=d.get("enable_alignment_flagger"),
            warning_threshold=d.get("alignment_warning_threshold"),
            failure_threshold=d.get("alignment_failure_threshold"),
            improvement_warning_threshold=d.get("improvement_warning_threshold"),
            improvement_failure_threshold=d.get("improvement_failure_threshold"),
            baseline_metrics_path=d.get("alignment_baseline_metrics_path"),
        )
        d["auto_merge"] = AutoMergeSettings(
            roi_threshold=d.get("auto_merge_roi_threshold"),
            coverage_threshold=d.get("auto_merge_coverage_threshold"),
        )
        d["actor_critic"] = ActorCriticSettings(
            actor_lr=d.get("ac_actor_lr"),
            critic_lr=d.get("ac_critic_lr"),
            gamma=d.get("ac_gamma"),
            epsilon=d.get("ac_epsilon"),
            epsilon_decay=d.get("ac_epsilon_decay"),
            buffer_size=d.get("ac_buffer_size"),
            batch_size=d.get("ac_batch_size"),
            checkpoint_path=d.get("ac_checkpoint_path"),
            normalize_states=d.get("ac_normalize_states"),
            reward_scale=d.get("ac_reward_scale"),
            eval_interval=d.get("ac_eval_interval"),
            checkpoint_interval=d.get("ac_checkpoint_interval"),
        )
        d["policy"] = PolicySettings(
            alpha=d.get("policy_alpha"),
            gamma=d.get("policy_gamma"),
            epsilon=d.get("policy_epsilon"),
            temperature=d.get("policy_temperature"),
            exploration=d.get("policy_exploration"),
        )

    def model_dump(self, mode: str = "python") -> dict[str, Any]:  # pragma: no cover - trivial
        return _deep_copy(self._data)

    def dict(self) -> dict[str, Any]:  # pragma: no cover - alias
        return self.model_dump()

    def __getattr__(self, name: str) -> Any:
        try:
            return self._data[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__slots__:
            super().__setattr__(name, value)
        else:
            self._data[name] = value


@lru_cache(maxsize=1)
def load_sandbox_settings(path: str | None = None) -> SandboxSettings:
    """Load :class:`SandboxSettings` from an optional YAML or JSON file."""

    data: dict[str, Any] = {}
    if path:
        resolved = resolve_path(path)
        with open(resolved, "r", encoding="utf-8") as fh:
            if resolved.suffix in (".yml", ".yaml"):
                if yaml is None:
                    missing = (
                        getattr(_MISSING_YAML, "name", "PyYAML")
                        if _MISSING_YAML
                        else "PyYAML"
                    )
                    raise ModuleNotFoundError(
                        f"{missing} is required to load YAML sandbox settings"
                    ) from _MISSING_YAML
                data = yaml.safe_load(fh) or {}
            elif resolved.suffix == ".json":
                data = json.load(fh)
            else:  # pragma: no cover - defensive
                raise ValueError(f"Unsupported config format: {resolved}")
    return SandboxSettings(**data)


__all__ = [
    "SandboxSettings",
    "AlignmentRules",
    "ROISettings",
    "SynergySettings",
    "BotThresholds",
    "AlignmentSettings",
    "load_sandbox_settings",
    "normalize_workflow_tests",
    "DEFAULT_SEVERITY_SCORE_MAP",
    "USING_SANDBOX_SETTINGS_FALLBACK",
]

