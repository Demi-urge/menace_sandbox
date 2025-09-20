from __future__ import annotations

"""Pydantic settings for sandbox utilities."""

import json
import os
from typing import Any
from pathlib import Path

import yaml
try:
    from .dynamic_path_router import resolve_path
except Exception:  # pragma: no cover
    from dynamic_path_router import resolve_path  # type: ignore

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict

    PYDANTIC_V2 = True
except Exception:  # pragma: no cover - fallback for pydantic<2
    from pydantic import BaseSettings  # type: ignore

    PYDANTIC_V2 = False
    SettingsConfigDict = dict  # type: ignore[misc]
from pydantic import BaseModel, Field

try:  # pragma: no cover - compatibility shim
    from pydantic import field_validator
except Exception:  # pragma: no cover
    from pydantic import validator as _pydantic_validator  # type: ignore

    def field_validator(*fields, **kwargs):  # type: ignore[misc]
        """Shim for ``pydantic.validator`` with ``allow_reuse`` defaulting to ``True``."""

        kwargs.setdefault("allow_reuse", True)
        return _pydantic_validator(*fields, **kwargs)

SELF_CODING_ROI_DROP: float = float(os.getenv("SELF_CODING_ROI_DROP", "-0.1"))
SELF_CODING_ERROR_INCREASE: float = float(
    os.getenv("SELF_CODING_ERROR_INCREASE", "1.0")
)

DEFAULT_SEVERITY_SCORE_MAP: dict[str, float] = {
    "critical": 100.0,
    "crit": 100.0,
    "fatal": 100.0,
    "high": 75.0,
    "error": 75.0,
    "warn": 50.0,
    "warning": 50.0,
    "medium": 50.0,
    "low": 25.0,
    "info": 0.0,
}


def normalize_workflow_tests(value: Any) -> list[str]:
    """Coerce arbitrary input into a list of workflow test selectors."""

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


class AlignmentRules(BaseModel):
    """Thresholds for human-alignment checks."""

    max_complexity_score: int = 10
    max_comment_density_drop: float = 0.1
    allowed_network_calls: int = 0
    comment_density_severity: int = 2
    network_call_severity: int = 3
    rule_modules: list[str] = Field(default_factory=list)


class ROISettings(BaseModel):
    """Settings related to return-on-investment scoring."""

    threshold: float | None = None
    confidence: float | None = None
    ema_alpha: float = 0.1
    compounding_weight: float = 1.0
    min_integration_roi: float = 0.0
    entropy_window: int = 5
    entropy_weight: float = 0.1
    entropy_threshold: float | None = None
    entropy_plateau_threshold: float | None = None
    entropy_plateau_consecutive: int | None = None
    entropy_ceiling_threshold: float | None = None
    entropy_ceiling_consecutive: int | None = None
    baseline_window: int = 5
    deviation_tolerance: float = 0.05
    stagnation_threshold: float = 0.01
    momentum_window: int = 5
    stagnation_cycles: int = 3
    momentum_dev_multiplier: float = 1.0
    roi_stagnation_dev_multiplier: float = 1.0

    @field_validator(
        "threshold",
        "confidence",
        "ema_alpha",
        "entropy_threshold",
        "entropy_plateau_threshold",
        "entropy_ceiling_threshold",
    )
    def _check_unit_range(cls, v: float | None, info: Any) -> float | None:
        if v is not None and not 0 <= v <= 1:
            raise ValueError(f"{info.field_name} must be between 0 and 1")
        return v

    @field_validator(
        "compounding_weight",
        "min_integration_roi",
        "entropy_weight",
    )
    def _check_non_negative(cls, v: float, info: Any) -> float:
        if v < 0:
            raise ValueError(f"{info.field_name} must be non-negative")
        return v

    @field_validator(
        "deviation_tolerance",
        "stagnation_threshold",
        "momentum_dev_multiplier",
        "roi_stagnation_dev_multiplier",
    )
    def _check_positive_float(cls, v: float, info: Any) -> float:
        if v <= 0:
            raise ValueError(f"{info.field_name} must be positive")
        return v

    @field_validator("entropy_plateau_consecutive", "entropy_ceiling_consecutive")
    def _check_positive(cls, v: int | None, info: Any) -> int | None:
        if v is not None and v <= 0:
            raise ValueError(f"{info.field_name} must be a positive integer")
        return v

    @field_validator(
        "baseline_window",
        "entropy_window",
        "stagnation_cycles",
        "momentum_window",
    )
    def _check_positive_int(cls, v: int, info: Any) -> int:
        if v <= 0:
            raise ValueError(f"{info.field_name} must be a positive integer")
        return v


class BotThresholds(BaseModel):
    """Per-bot ROI and error thresholds."""

    roi_drop: float | None = None
    error_threshold: float | None = None
    test_failure_threshold: float | None = None
    patch_success_drop: float | None = None
    test_command: list[str] | None = None
    workflow_tests: list[str] = Field(default_factory=list)

    if PYDANTIC_V2:

        @field_validator("workflow_tests", mode="before")
        def _validate_workflow_tests(cls, value: Any) -> list[str]:
            return normalize_workflow_tests(value)

    else:  # pragma: no cover - compatibility for pydantic<2

        @field_validator("workflow_tests", pre=True)
        def _validate_workflow_tests(cls, value: Any) -> list[str]:
            return normalize_workflow_tests(value)


class SynergySettings(BaseModel):
    """Settings for module synergy calculations."""

    threshold: float | None = None
    confidence: float | None = None
    threshold_window: int | None = None
    threshold_weight: float | None = None
    ma_window: int | None = None
    stationarity_confidence: float | None = None
    std_threshold: float | None = None
    variance_confidence: float | None = None
    weight_roi: float = 1.0
    weight_efficiency: float = 1.0
    weight_resilience: float = 1.0
    weight_antifragility: float = 1.0
    weight_reliability: float = 1.0
    weight_maintainability: float = 1.0
    weight_throughput: float = 1.0
    weights_lr: float = 0.1
    train_interval: int = 10
    replay_size: int = 100
    batch_size: int = 32
    gamma: float = 0.99
    noise: float = 0.1
    hidden_size: int = 32
    layers: int = 1
    optimizer: str = "adam"
    checkpoint_interval: int = 50
    strategy: str = "dqn"
    target_sync: int = 10
    python_fallback: bool = True
    python_max_replay: int = 1000
    deviation_tolerance: float = 0.0

    @field_validator(
        "threshold",
        "confidence",
        "threshold_weight",
        "stationarity_confidence",
        "std_threshold",
        "variance_confidence",
        "gamma",
    )
    def _synergy_unit_range(cls, v: float | None, info: Any) -> float | None:
        if v is not None and not 0 <= v <= 1:
            raise ValueError(f"{info.field_name} must be between 0 and 1")
        return v

    @field_validator(
        "threshold_window",
        "ma_window",
        "train_interval",
        "replay_size",
        "batch_size",
        "hidden_size",
        "layers",
        "checkpoint_interval",
        "target_sync",
        "python_max_replay",
    )
    def _synergy_positive_int(cls, v: int | None, info: Any) -> int | None:
        if v is not None and v <= 0:
            raise ValueError(f"{info.field_name} must be a positive integer")
        return v

    @field_validator(
        "weight_roi",
        "weight_efficiency",
        "weight_resilience",
        "weight_antifragility",
        "weight_reliability",
        "weight_maintainability",
        "weight_throughput",
        "weights_lr",
        "noise",
        "deviation_tolerance",
    )
    def _synergy_non_negative(cls, v: float, info: Any) -> float:
        if v < 0:
            raise ValueError(f"{info.field_name} must be non-negative")
        return v

    @field_validator("strategy")
    def _synergy_strategy(cls, v: str) -> str:
        allowed = {"dqn", "double_dqn", "sac", "td3"}
        if v not in allowed:
            raise ValueError(f"strategy must be one of {sorted(allowed)}")
        return v


class AlignmentSettings(BaseModel):
    """Grouping of alignment-related settings."""

    rules: AlignmentRules = Field(default_factory=AlignmentRules)
    enable_flagger: bool = True
    warning_threshold: float = 0.5
    failure_threshold: float = 0.9
    improvement_warning_threshold: float = 0.5
    improvement_failure_threshold: float = 0.9
    baseline_metrics_path: Path = Field(
        default_factory=lambda: resolve_path("sandbox_metrics.yaml")
    )

    @field_validator(
        "warning_threshold",
        "failure_threshold",
        "improvement_warning_threshold",
        "improvement_failure_threshold",
    )
    def _alignment_unit_range(cls, v: float, info: Any) -> float:
        if not 0 <= v <= 1:
            raise ValueError(f"{info.field_name} must be between 0 and 1")
        return v


class AutoMergeSettings(BaseModel):
    """Thresholds controlling automatic merges."""

    roi_threshold: float = 0.0
    coverage_threshold: float = 1.0

    @field_validator("roi_threshold")
    def _roi_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("roi_threshold must be non-negative")
        return v

    @field_validator("coverage_threshold")
    def _cov_unit_range(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("coverage_threshold must be between 0 and 1")
        return v


class ActorCriticSettings(BaseModel):
    """Hyperparameters for the actor-critic learning agent."""

    actor_lr: float = 0.01
    critic_lr: float = 0.02
    gamma: float = 0.95
    epsilon: float = 0.1
    epsilon_decay: float = 0.99
    buffer_size: int = 100
    batch_size: int = 32
    checkpoint_path: str = "actor_critic_state.json"
    normalize_states: bool = True
    reward_scale: float = 1.0
    eval_interval: int = 100
    checkpoint_interval: int = 100

    @field_validator(
        "actor_lr",
        "critic_lr",
        "gamma",
        "epsilon",
        "epsilon_decay",
    )
    def _ac_unit_range(cls, v: float, info: Any) -> float:
        if v <= 0:
            raise ValueError(f"{info.field_name} must be positive")
        if info.field_name in {"gamma", "epsilon", "epsilon_decay"} and v > 1:
            raise ValueError(f"{info.field_name} must be between 0 and 1")
        return v

    @field_validator("buffer_size", "batch_size")
    def _ac_positive_int(cls, v: int, info: Any) -> int:
        if v <= 0:
            raise ValueError(f"{info.field_name} must be a positive integer")
        return v

    @field_validator("reward_scale")
    def _ac_positive_float(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("reward_scale must be positive")
        return v

    @field_validator("eval_interval", "checkpoint_interval")
    def _ac_interval(cls, v: int, info: Any) -> int:
        if v <= 0:
            raise ValueError(f"{info.field_name} must be a positive integer")
        return v


class PolicySettings(BaseModel):
    """Default hyperparameters for self-improvement policies."""

    alpha: float = 0.5
    gamma: float = 0.9
    epsilon: float = 0.1
    temperature: float = 1.0
    exploration: str = "epsilon_greedy"

    @field_validator("alpha", "gamma", "epsilon")
    def _policy_unit_range(cls, v: float, info: Any) -> float:
        if not 0 <= v <= 1:
            raise ValueError(f"{info.field_name} must be between 0 and 1")
        return v

    @field_validator("temperature")
    def _policy_temperature(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("temperature must be positive")
        return v


class SandboxSettings(BaseSettings):
    """Environment configuration for sandbox runners.

    Provides Codex retry and fallback controls via ``codex_retry_delays`` and
    ``codex_fallback_model`` which map to the ``CODEX_RETRY_DELAYS`` and
    ``CODEX_FALLBACK_MODEL`` environment variables, respectively. Prompt
    simplification is tunable with ``simplify_prompt_drop_system`` and
    ``simplify_prompt_example_limit``.
    """

    menace_mode: str = Field("test", env="MENACE_MODE")
    database_url: str = Field("", env="DATABASE_URL")
    menace_offline_install: bool = Field(False, env="MENACE_OFFLINE_INSTALL")
    menace_wheel_dir: str | None = Field(None, env="MENACE_WHEEL_DIR")
    menace_light_imports: bool = Field(False, env="MENACE_LIGHT_IMPORTS")
    auto_install_dependencies: bool = Field(
        False,
        env="AUTO_INSTALL_DEPENDENCIES",
        description="Automatically install missing Python packages when possible.",
    )
    inject_edge_cases: bool = Field(True, env="INJECT_EDGE_CASES")
    roi_cycles: int | None = Field(None, env="ROI_CYCLES")
    synergy_cycles: int | None = Field(None, env="SYNERGY_CYCLES")
    baseline_window: int = Field(10, env="BASELINE_WINDOW")
    adaptive_thresholds: bool = Field(
        False, env="ADAPTIVE_THRESHOLDS"
    )
    mae_deviation: float = Field(1.0, env="MAE_DEVIATION")
    acc_deviation: float = Field(1.0, env="ACC_DEVIATION")
    energy_deviation: float = Field(1.0, env="ENERGY_DEVIATION")
    roi_deviation: float = Field(1.0, env="ROI_DEVIATION")
    entropy_deviation: float = Field(1.0, env="ENTROPY_DEVIATION")
    error_overfit_percentile: float = Field(
        0.95, env="ERROR_OVERFIT_PERCENTILE"
    )
    entropy_overfit_percentile: float = Field(
        0.95, env="ENTROPY_OVERFIT_PERCENTILE"
    )
    autoscale_create_dev_multiplier: float = Field(
        0.8, env="AUTOSCALE_CREATE_DEV_MULTIPLIER"
    )
    autoscale_remove_dev_multiplier: float = Field(
        0.3, env="AUTOSCALE_REMOVE_DEV_MULTIPLIER"
    )
    autoscale_roi_dev_multiplier: float = Field(
        0.0, env="AUTOSCALE_ROI_DEV_MULTIPLIER"
    )
    scenario_alert_dev_multiplier: float = Field(
        1.0, env="SCENARIO_ALERT_DEV_MULTIPLIER"
    )
    scenario_patch_dev_multiplier: float = Field(
        2.0, env="SCENARIO_PATCH_DEV_MULTIPLIER"
    )
    scenario_rerun_dev_multiplier: float = Field(
        3.0, env="SCENARIO_RERUN_DEV_MULTIPLIER"
    )
    save_synergy_history: bool | None = Field(None, env="SAVE_SYNERGY_HISTORY")
    severity_score_map: dict[str, float] = Field(
        default_factory=lambda: DEFAULT_SEVERITY_SCORE_MAP.copy(),
        env="SEVERITY_SCORE_MAP",
        description=(
            "Mapping of error severities to numeric scores used during cycle "
            "evaluation. Defaults to "
            f"{DEFAULT_SEVERITY_SCORE_MAP}."
        ),
    )

    @field_validator("baseline_window")
    def _baseline_window_range(cls, v: int) -> int:
        if not 5 <= v <= 10:
            raise ValueError("baseline_window must be between 5 and 10")
        return v

    @field_validator("relevancy_history_min_length")
    def _relevancy_history_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("relevancy_history_min_length must be non-negative")
        return v

    @field_validator("roi_baseline_window")
    def _roi_baseline_window_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("roi_baseline_window must be positive")
        return v

    @field_validator("roi_momentum_window")
    def _roi_momentum_window_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("roi_momentum_window must be positive")
        return v

    @field_validator("roi_stagnation_cycles")
    def _roi_stagnation_cycles_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("roi_stagnation_cycles must be positive")
        return v

    @field_validator(
        "roi_deviation_tolerance",
        "roi_stagnation_threshold",
        "roi_momentum_dev_multiplier",
    )
    def _roi_positive_float(cls, v: float, info: Any) -> float:
        if v <= 0:
            raise ValueError(f"{info.field_name} must be positive")
        return v

    @field_validator("error_overfit_percentile", "entropy_overfit_percentile")
    def _overfit_percentile_range(cls, v: float, info: Any) -> float:
        if not 0 < v <= 1:
            raise ValueError(f"{info.field_name} must be between 0 and 1")
        return v

    @field_validator(
        "mae_deviation",
        "acc_deviation",
        "energy_deviation",
        "roi_deviation",
        "entropy_deviation",
        "autoscale_create_dev_multiplier",
        "autoscale_remove_dev_multiplier",
        "autoscale_roi_dev_multiplier",
        "relevancy_deviation_multiplier",
        "scenario_alert_dev_multiplier",
        "scenario_patch_dev_multiplier",
        "scenario_rerun_dev_multiplier",
    )
    def _baseline_non_negative(cls, v: float, info: Any) -> float:
        if v < 0:
            raise ValueError(f"{info.field_name} must be non-negative")
        return v
    menace_env_file: str = Field(
        (resolve_path(".") / ".env").as_posix(), env="MENACE_ENV_FILE"
    )
    sandbox_data_dir: str = Field(
        resolve_path("sandbox_data").as_posix(), env="SANDBOX_DATA_DIR"
    )
    sandbox_env_presets: str | None = Field(None, env="SANDBOX_ENV_PRESETS")
    required_env_vars: list[str] = Field(
        default_factory=lambda: [
            "OPENAI_API_KEY",
            "DATABASE_URL",
            "MODELS",
        ],  # Stripe keys are handled via stripe_billing_router
        env="SANDBOX_REQUIRED_ENV_VARS",
        description="Environment variables required for sandbox initialisation.",
    )
    sandbox_required_db_files: list[str] = Field(
        default_factory=lambda: [
            "metrics.db",
            "patch_history.db",
        ],
        env="SANDBOX_REQUIRED_DB_FILES",
        description="SQLite database files expected in the sandbox data directory.",
    )
    sandbox_repo_path: str = Field(
        resolve_path(".").as_posix(),
        env="SANDBOX_REPO_PATH",
        description="Path to repository root for sandbox operations.",
    )
    sandbox_backend: str = Field(
        "venv",
        env="SANDBOX_BACKEND",
        description="Sandbox execution backend: 'venv' or 'docker'.",
    )
    sandbox_docker_image: str = Field(
        "python:3.11-slim",
        env="SANDBOX_DOCKER_IMAGE",
        description="Docker image for sandbox runs when using the docker backend.",
    )
    sandbox_central_logging: bool = Field(
        True,
        env="SANDBOX_CENTRAL_LOGGING",
        description="Enable centralised logging output.",
    )
    log_rotation_max_bytes: int = Field(
        5 * 1024 * 1024,
        env="LOG_ROTATION_MAX_BYTES",
        ge=0,
        description=(
            "Rotate logs when file exceeds this size in bytes; "
            "0 disables size-based rotation."
        ),
    )
    log_rotation_backup_count: int = Field(
        5,
        env="LOG_ROTATION_BACKUP_COUNT",
        ge=0,
        description="Number of rotated log files to keep.",
    )
    log_rotation_seconds: int | None = Field(
        None,
        env="LOG_ROTATION_SECONDS",
        gt=0,
        description="Rotate logs after this many seconds; when set, overrides size-based rotation.",
    )
    local_knowledge_refresh_interval: float = Field(
        600.0, env="LOCAL_KNOWLEDGE_REFRESH_INTERVAL"
    )
    menace_local_db_path: str | None = Field(None, env="MENACE_LOCAL_DB_PATH")
    menace_shared_db_path: str = Field(
        (resolve_path(".") / "shared" / "global.db").as_posix(),
        env="MENACE_SHARED_DB_PATH",
    )
    sandbox_volatility_threshold: float = Field(1.0, env="SANDBOX_VOLATILITY_THRESHOLD")
    gpt_memory_compact_interval: float | None = Field(
        None, env="GPT_MEMORY_COMPACT_INTERVAL"
    )
    export_synergy_metrics: bool = Field(False, env="EXPORT_SYNERGY_METRICS")
    synergy_metrics_port: int = Field(8003, env="SYNERGY_METRICS_PORT")
    metrics_port: int | None = Field(None, env="METRICS_PORT")
    relevancy_radar_interval: float | None = Field(None, env="RELEVANCY_RADAR_INTERVAL")
    sandbox_generate_presets: bool = Field(True, env="SANDBOX_GENERATE_PRESETS")
    sandbox_module_algo: str | None = Field(None, env="SANDBOX_MODULE_ALGO")
    sandbox_module_threshold: float | None = Field(None, env="SANDBOX_MODULE_THRESHOLD")
    sandbox_semantic_modules: bool = Field(False, env="SANDBOX_SEMANTIC_MODULES")
    sandbox_stub_model: str | None = Field(None, env="SANDBOX_STUB_MODEL")
    risk_weight_commit: float = Field(0.2, env="RISK_WEIGHT_COMMIT")
    risk_weight_complexity: float = Field(0.4, env="RISK_WEIGHT_COMPLEXITY")
    risk_weight_failures: float = Field(0.4, env="RISK_WEIGHT_FAILURES")
    llm_backend: str = Field("openai", env="LLM_BACKEND")
    llm_fallback_backend: str | None = Field(None, env="LLM_FALLBACK_BACKEND")
    preferred_llm_backend: str = Field(
        "openai",
        env="PREFERRED_LLM_BACKEND",
        description="Primary LLM backend to use when multiple are available.",
    )
    available_backends: dict[str, str] = Field(
        default_factory=lambda: {
            "openai": "llm_interface.OpenAIProvider",
            "anthropic": "anthropic_client.AnthropicClient",
            "ollama": "local_client.OllamaClient",
            "vllm": "local_client.VLLMClient",
            "mixtral": "local_backend.mixtral_client",
            "llama3": "local_backend.llama3_client",
            "private": "private_backend.local_weights_client",
        },
        env="AVAILABLE_LLM_BACKENDS",
        description=(
            "Mapping of backend names to import paths for LLM client factories."
            " Entries are automatically registered with llm_registry."
        ),
    )
    huggingface_token: str | None = Field(
        None, env=["HUGGINGFACE_API_TOKEN", "HF_TOKEN"]
    )
    sandbox_max_recursion_depth: int | None = Field(
        None, env="SANDBOX_MAX_RECURSION_DEPTH"
    )
    sandbox_log_level: str = Field("INFO", env="SANDBOX_LOG_LEVEL")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    sandbox_retry_delay: float = Field(
        1.0,
        env="SANDBOX_RETRY_DELAY",
        description="Initial delay between restart attempts in seconds.",
    )
    sandbox_retry_backoff_multiplier: float = Field(
        1.0,
        env="SANDBOX_RETRY_BACKOFF_MULTIPLIER",
        description="Multiplier applied to the base delay for each retry attempt.",
    )
    sandbox_retry_jitter: float = Field(
        0.0,
        env="SANDBOX_RETRY_JITTER",
        description="Maximum jitter in seconds added to retry delays.",
    )
    sandbox_max_retries: int | None = Field(
        None,
        env="SANDBOX_MAX_RETRIES",
        description="Maximum number of restart attempts before giving up.",
    )
    sandbox_circuit_max_failures: int = Field(
        5,
        env="SANDBOX_CIRCUIT_MAX_FAILURES",
        description="Consecutive failures allowed before opening the circuit breaker.",
    )
    sandbox_circuit_reset_timeout: float = Field(
        60.0,
        env="SANDBOX_CIRCUIT_RESET_TIMEOUT",
        description="Time in seconds before a tripped circuit resets.",
    )
    required_system_tools: list[str] = Field(
        default_factory=lambda: ["ffmpeg", "tesseract", "qemu-system-x86_64"],
        env="SANDBOX_REQUIRED_SYSTEM_TOOLS",
        description="System commands expected to be available on PATH.",
    )
    required_python_packages: list[str] = Field(
        default_factory=lambda: [
            "pydantic",
            "dotenv",
            "foresight_tracker",
            "filelock",
        ],
        env="SANDBOX_REQUIRED_PYTHON_PKGS",
        description="Python packages required for sandbox operation.",
    )
    optional_python_packages: list[str] = Field(
        default_factory=lambda: [
            "matplotlib",
            "statsmodels",
            "uvicorn",
            "fastapi",
            "sklearn",
            "httpx",
        ],
        env="SANDBOX_OPTIONAL_PYTHON_PKGS",
        description="Optional Python packages used by sandbox utilities.",
    )
    install_optional_dependencies: bool = Field(
        False,
        env="INSTALL_OPTIONAL_DEPENDENCIES",
        description="Automatically attempt to install missing optional modules.",
    )
    retry_optional_dependencies: bool = Field(
        False,
        env="RETRY_OPTIONAL_DEPENDENCIES",
        description="Retry loading optional modules instead of using a stub.",
    )
    optional_service_versions: dict[str, str] = Field(
        default_factory=lambda: {
            "relevancy_radar": "1.0.0",
            "quick_fix_engine": "1.0.0",
        },
        env="OPTIONAL_SERVICE_VERSIONS",
        description="Mapping of optional service modules to minimum versions.",
    )
    if PYDANTIC_V2:

        @field_validator(
            "required_system_tools",
            "required_python_packages",
            "optional_python_packages",
            "required_env_vars",
            "sandbox_required_db_files",
            mode="before",
        )
        def _parse_dependency_list(cls, v: Any) -> Any:
            if isinstance(v, str):
                try:
                    parsed = json.loads(v)
                    if isinstance(parsed, list):
                        return parsed
                except Exception:
                    return [i.strip() for i in v.split(",") if i.strip()]
            return v

        @field_validator("optional_service_versions", mode="before")
        def _parse_optional_service_versions(cls, v: Any) -> Any:
            if isinstance(v, str):
                try:
                    return json.loads(v) if v else {}
                except Exception:
                    return {}
            return v

        @field_validator("available_backends", mode="before")
        def _parse_available_backends(cls, v: Any) -> Any:
            if isinstance(v, str):
                try:
                    return json.loads(v) if v else {}
                except Exception:
                    return {}
            return v
    else:  # pragma: no cover - pydantic<2

        @field_validator(
            "required_system_tools",
            "required_python_packages",
            "optional_python_packages",
            "required_env_vars",
            "sandbox_required_db_files",
            pre=True,
        )
        def _parse_dependency_list(cls, v: Any) -> Any:  # type: ignore[override]
            if isinstance(v, str):
                try:
                    parsed = json.loads(v)
                    if isinstance(parsed, list):
                        return parsed
                except Exception:
                    return [i.strip() for i in v.split(",") if i.strip()]
            return v

        @field_validator("optional_service_versions", pre=True)
        def _parse_optional_service_versions(cls, v: Any) -> Any:  # type: ignore[override]
            if isinstance(v, str):
                try:
                    return json.loads(v) if v else {}
                except Exception:
                    return {}
            return v

        @field_validator("available_backends", pre=True)
        def _parse_available_backends(cls, v: Any) -> Any:  # type: ignore[override]
            if isinstance(v, str):
                try:
                    return json.loads(v) if v else {}
                except Exception:
                    return {}
            return v
    openai_api_key: str | None = Field(
        None, env="OPENAI_API_KEY", description="API key for OpenAI access."
    )
    codex_retry_delays: list[int] = Field(
        default_factory=lambda: [2, 5, 10],
        env="CODEX_RETRY_DELAYS",
        description="Retry delays in seconds for Codex API calls.",
    )
    codex_timeout: float = Field(
        30.0,
        env="CODEX_TIMEOUT",
        description="Timeout in seconds for Codex API calls.",
    )
    codex_fallback_model: str = Field(
        "gpt-3.5-turbo",
        env="CODEX_FALLBACK_MODEL",
        description="Fallback model to use when Codex requests fail.",
    )
    codex_fallback_strategy: str = Field(
        "queue",
        env="CODEX_FALLBACK_STRATEGY",
        description="Fallback handling strategy: 'queue' or 'reroute'.",
    )
    codex_retry_queue_path: str = Field(
        "codex_retry_queue.jsonl",
        env="CODEX_RETRY_QUEUE",
        description="Path for the Codex retry queue file.",
    )
    simplify_prompt_drop_system: bool = Field(
        True,
        env="SIMPLIFY_PROMPT_DROP_SYSTEM",
        description="Remove system message when simplifying prompts.",
    )
    simplify_prompt_example_limit: int | None = Field(
        0,
        env="SIMPLIFY_PROMPT_EXAMPLE_LIMIT",
        description=(
            "Maximum number of few-shot examples to retain when simplifying prompts. "
            "0 drops all examples; None keeps all."
        ),
    )

    @field_validator("codex_fallback_strategy")
    def _codex_strategy_valid(cls, v: str) -> str:
        if v not in {"queue", "reroute"}:
            raise ValueError("codex_fallback_strategy must be 'queue' or 'reroute'")
        return v

    if PYDANTIC_V2:

        @field_validator("codex_retry_delays", mode="before")
        def _parse_codex_retry_delays(cls, v: Any) -> Any:
            if isinstance(v, str):
                try:
                    parsed = json.loads(v)
                    if isinstance(parsed, list):
                        return [int(i) for i in parsed]
                except Exception:
                    return [int(i.strip()) for i in v.split(",") if i.strip()]
            return v

    else:  # pragma: no cover - pydantic<2

        @field_validator("codex_retry_delays", pre=True)
        def _parse_codex_retry_delays(cls, v: Any) -> Any:  # type: ignore[override]
            if isinstance(v, str):
                try:
                    parsed = json.loads(v)
                    if isinstance(parsed, list):
                        return [int(i) for i in parsed]
                except Exception:
                    return [int(i.strip()) for i in v.split(",") if i.strip()]
            return v
    audit_log_path: str = Field(
        "audit.log",
        env="AUDIT_LOG_PATH",
        description="Path where the audit trail is written.",
    )
    audit_privkey: str | None = Field(
        None,
        env="AUDIT_PRIVKEY",
        description="Base64-encoded private key for signing audit entries.",
    )
    prompt_success_log_path: str = Field(
        "sandbox_data/prompt_success_log.jsonl",
        env="PROMPT_SUCCESS_LOG_PATH",
        description="Path for recording successful prompt executions.",
    )
    prompt_failure_log_path: str = Field(
        "sandbox_data/prompt_failure_log.jsonl",
        env="PROMPT_FAILURE_LOG_PATH",
        description="Path for recording failed prompt executions.",
    )
    prompt_penalty_path: str = Field(
        "sandbox_data/prompt_penalties.json",
        env="PROMPT_PENALTY_PATH",
        description="File storing per-prompt regression counts.",
    )
    prompt_failure_threshold: int = Field(
        3,
        env="PROMPT_FAILURE_THRESHOLD",
        description="Failure count after which prompt selection is penalised.",
    )
    prompt_penalty_multiplier: float = Field(
        0.1,
        env="PROMPT_PENALTY_MULTIPLIER",
        description="Multiplier applied to value estimates of penalised prompts.",
    )
    prompt_roi_decay_rate: float = Field(
        0.0,
        env="PROMPT_ROI_DECAY_RATE",
        description="Exponential decay rate applied to prompt ROI history.",
    )
    strategy_failure_limits: dict[str, int] = Field(
        default_factory=dict,
        env="STRATEGY_FAILURE_LIMITS",
        description="Consecutive failure limits per strategy before rotation.",
    )

    @field_validator("strategy_failure_limits", mode="before")
    def _parse_strategy_failure_limits(cls, v: Any) -> Any:
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception:
                return {}
        return v

    @field_validator("prompt_roi_decay_rate")
    def _check_non_negative_decay(cls, v: float) -> float:
        if v < 0:
            raise ValueError("prompt_roi_decay_rate must be non-negative")
        return v
    failure_fingerprint_path: str = Field(
        "failure_fingerprints.jsonl",
        env="FAILURE_FINGERPRINT_PATH",
        description="Path for storing failure fingerprint records.",
    )
    fingerprint_similarity_threshold: float = Field(
        0.8,
        env="FINGERPRINT_SIMILARITY_THRESHOLD",
        description="Default cosine similarity threshold for matching failure fingerprints.",
    )
    prompt_chunk_token_threshold: int = Field(
        3500,
        env="PROMPT_CHUNK_TOKEN_THRESHOLD",
        description=(
            "Token limit for individual code chunks handled by the consolidated "
            "chunking helpers. Falls back to naive line splitting when tokenisation "
            "is unavailable."
        ),
    )
    chunk_token_threshold: int = Field(
        3500,
        env="CHUNK_TOKEN_THRESHOLD",
        description="Token limit for chunked prompting operations after summarisation.",
    )
    chunk_summary_cache_dir: Path = Field(
        Path("chunk_summary_cache"),
        env=["CHUNK_SUMMARY_CACHE_DIR", "PROMPT_CHUNK_CACHE_DIR"],
        description=(
            "Directory for cached summaries. Remove to force regeneration or relocate "
            "with an environment variable."
        ),
    )

    @property
    def prompt_chunk_cache_dir(self) -> Path:  # pragma: no cover - backward compat
        """Alias for :attr:`chunk_summary_cache_dir`.

        Historically the settings exposed ``prompt_chunk_cache_dir`` which
        matched the environment variable ``PROMPT_CHUNK_CACHE_DIR``.  The new
        field :attr:`chunk_summary_cache_dir` supersedes it but this property is
        retained so older code can continue to access the cache directory via
        the previous attribute name.
        """

        return self.chunk_summary_cache_dir
    stub_timeout: float = Field(
        10.0,
        env="SANDBOX_STUB_TIMEOUT",
        description="Timeout for stub generation in seconds.",
    )
    stub_save_timeout: float = Field(
        5.0,
        env="SANDBOX_STUB_SAVE_TIMEOUT",
        description="Maximum time to await pending stub cache save tasks.",
    )
    stub_retries: int = Field(
        2,
        env="SANDBOX_STUB_RETRIES",
        description="Retry attempts for stub generation.",
    )
    stub_retry_base: float = Field(
        0.5,
        env="SANDBOX_STUB_RETRY_BASE",
        description="Initial delay for exponential backoff in seconds.",
    )
    stub_retry_max: float = Field(
        30.0,
        env="SANDBOX_STUB_RETRY_MAX",
        description="Maximum delay for exponential backoff in seconds.",
    )
    stub_cache_max: int = Field(
        1024,
        env="SANDBOX_STUB_CACHE_MAX",
        description="Maximum number of stub responses cached.",
    )
    stub_fallback_model: str = Field(
        "distilgpt2",
        env="SANDBOX_STUB_FALLBACK_MODEL",
        description="Model name to fall back to when the preferred generator fails.",
    )
    stub_providers: list[str] | None = Field(
        None,
        env="SANDBOX_STUB_PROVIDERS",
        description=(
            "Comma-separated stub provider names to enable. When unset, all "
            "discovered providers are used."
        ),
    )
    disabled_stub_providers: list[str] = Field(
        default_factory=list,
        env="SANDBOX_DISABLED_STUB_PROVIDERS",
        description="Comma-separated stub provider names to disable.",
    )
    if PYDANTIC_V2:

        @field_validator("stub_providers", "disabled_stub_providers", mode="before")
        def _split_stub_providers(cls, v: Any) -> Any:
            if isinstance(v, str):
                return [s.strip() for s in v.split(",") if s.strip()]
            return v

    else:  # pragma: no cover - pydantic<2

        @field_validator("stub_providers", "disabled_stub_providers", pre=True)
        def _split_stub_providers(cls, v: Any) -> Any:  # type: ignore[override]
            if isinstance(v, str):
                return [s.strip() for s in v.split(",") if s.strip()]
            return v

    stub_strategy: str | None = Field(
        None,
        env="SANDBOX_STUB_STRATEGY",
        description="Primary stub generation strategy to attempt first.",
    )
    stub_strategy_order: list[str] | None = Field(
        None,
        env="SANDBOX_STUB_STRATEGY_ORDER",
        description="Comma-separated fallback order for stub strategies.",
    )
    input_templates_file: str = Field(
        (resolve_path("sandbox_data") / "input_stub_templates.json").as_posix(),
        env="SANDBOX_INPUT_TEMPLATES_FILE",
        description="Path to input stub templates JSON file.",
    )
    input_history: str | None = Field(
        None,
        env="SANDBOX_INPUT_HISTORY",
        description="Path to input history database or JSONL file.",
    )
    stub_seed: int | None = Field(
        None,
        env="SANDBOX_STUB_SEED",
        description="Seed value for deterministic stub generation.",
    )
    stub_random_config: dict[str, Any] = Field(
        default_factory=dict,
        env="SANDBOX_STUB_RANDOM_CONFIG",
        description="JSON-encoded configuration for the random strategy.",
    )
    stub_random_config_file: str | None = Field(
        None,
        env="SANDBOX_RANDOM_CONFIG_FILE",
        description="Path to JSON/YAML file providing random strategy parameters.",
    )
    stub_random_generator: str | None = Field(
        None,
        env="SANDBOX_RANDOM_GENERATOR",
        description="Import path to callable producing random input stubs.",
    )
    misuse_stubs: bool = Field(
        False,
        env="SANDBOX_MISUSE_STUBS",
        description="Append misuse stubs to generated inputs when true.",
    )
    if PYDANTIC_V2:

        @field_validator("stub_strategy_order", mode="before")
        def _split_stub_strategy_order(cls, v: Any) -> Any:
            if isinstance(v, str):
                return [s.strip() for s in v.split(",") if s.strip()]
            return v

        @field_validator("stub_random_config", mode="before")
        def _parse_stub_random_config(cls, v: Any) -> Any:
            if isinstance(v, str):
                try:
                    return json.loads(v) if v else {}
                except Exception:
                    return {}
            return v

        @field_validator("stub_random_config_file", "stub_random_generator", mode="before")
        def _clean_random_fields(cls, v: Any) -> Any:
            if isinstance(v, str) and not v.strip():
                return None
            return v

        @field_validator("input_history", mode="before")
        def _empty_history(cls, v: Any) -> Any:
            if isinstance(v, str) and not v.strip():
                return None
            return v

    else:  # pragma: no cover - pydantic<2

        @field_validator("stub_strategy_order", pre=True)
        def _split_stub_strategy_order(cls, v: Any) -> Any:  # type: ignore[override]
            if isinstance(v, str):
                return [s.strip() for s in v.split(",") if s.strip()]
            return v

        @field_validator("stub_random_config", pre=True)
        def _parse_stub_random_config(cls, v: Any) -> Any:  # type: ignore[override]
            if isinstance(v, str):
                try:
                    return json.loads(v) if v else {}
                except Exception:
                    return {}
            return v

        @field_validator("stub_random_config_file", "stub_random_generator", pre=True)
        def _clean_random_fields(cls, v: Any) -> Any:  # type: ignore[override]
            if isinstance(v, str) and not v.strip():
                return None
            return v

        @field_validator("input_history", pre=True)
        def _empty_history(cls, v: Any) -> Any:  # type: ignore[override]
            if isinstance(v, str) and not v.strip():
                return None
            return v

    suggestion_sources: list[str] = Field(
        default_factory=lambda: ["cache", "knowledge", "heuristic"],
        env="SANDBOX_SUGGESTION_SOURCES",
        description=(
            "Comma-separated suggestion source order for offline patch hints."
        ),
    )
    if PYDANTIC_V2:

        @field_validator("suggestion_sources", mode="before")
        def _split_suggestion_sources(cls, v: Any) -> Any:
            if isinstance(v, str):
                return [s.strip() for s in v.split(",") if s.strip()]
            return v

    else:  # pragma: no cover - pydantic<2

        @field_validator("suggestion_sources", pre=True)
        def _split_suggestion_sources(cls, v: Any) -> Any:  # type: ignore[override]
            if isinstance(v, str):
                return [s.strip() for s in v.split(",") if s.strip()]
            return v

    meta_planning_interval: int = Field(
        10,
        env="META_PLANNING_INTERVAL",
        description="Cycles between meta planning runs.",
    )
    meta_planning_period: int = Field(
        3600,
        env="META_PLANNING_PERIOD",
        description="Seconds between background meta planning runs.",
    )
    meta_planning_loop: bool = Field(
        False,
        env="META_PLANNING_LOOP",
        description="Run meta planning continuously in its own loop.",
    )
    enable_meta_planner: bool = Field(
        False,
        env="ENABLE_META_PLANNER",
        description="Fail fast if MetaWorkflowPlanner is unavailable.",
    )
    meta_improvement_threshold: float = Field(
        0.01,
        env="META_IMPROVEMENT_THRESHOLD",
        description="Minimum improvement required to accept meta plan updates.",
    )
    meta_mutation_rate: float = Field(
        1.0,
        env="META_MUTATION_RATE",
        description="Mutation rate multiplier for meta planning.",
    )
    meta_roi_weight: float = Field(
        1.0,
        env="META_ROI_WEIGHT",
        description="Weight applied to ROI when composing workflows.",
    )
    meta_domain_penalty: float = Field(
        1.0,
        env="META_DOMAIN_PENALTY",
        description="Penalty for domain transitions in meta planning.",
    )
    overfitting_entropy_threshold: float = Field(
        0.2,
        env="OVERFITTING_ENTROPY_THRESHOLD",
        description="Entropy delta triggering overfitting fallback.",
    )
    meta_entropy_threshold: float | None = Field(
        0.2,
        env="META_ENTROPY_THRESHOLD",
        description=(
            "Maximum allowed workflow entropy when recording improvements."
            " Defaults to 0.2 when unset."
        ),
    )
    meta_search_depth: int = Field(
        3,
        env="META_SEARCH_DEPTH",
        description="Maximum depth explored by fallback planner heuristic search.",
    )
    meta_beam_width: int = Field(
        5,
        env="META_BEAM_WIDTH",
        description="Number of top candidates kept during heuristic search.",
    )
    meta_entropy_weight: float = Field(
        0.0,
        env="META_ENTROPY_WEIGHT",
        description="Weight penalising workflow entropy when scoring chains.",
    )
    workflows_db: str = Field(
        (resolve_path(".") / "workflows.db").as_posix(),
        env="WORKFLOWS_DB",
        description="SQLite database storing workflow definitions.",
    )
    gpt_memory_db: str = Field(
        (resolve_path(".") / "gpt_memory.db").as_posix(),
        env="GPT_MEMORY_DB",
        description="Path to GPT memory database file.",
    )
    prune_interval: int = Field(
        50,
        env="PRUNE_INTERVAL",
        description="Number of GPT memory interactions before compaction.",
    )
    self_learning_eval_interval: int = Field(
        0,
        env="SELF_LEARNING_EVAL_INTERVAL",
        description="Training steps between evaluation passes.",
    )
    self_learning_summary_interval: int = Field(
        0,
        env="SELF_LEARNING_SUMMARY_INTERVAL",
        description="Training steps between summary logs.",
    )
    self_test_lock_file: str = Field(
        (resolve_path("sandbox_data") / "self_test.lock").as_posix(),
        env="SELF_TEST_LOCK_FILE",
        description="File used to serialise self-test runs.",
    )
    self_test_report_dir: str = Field(
        (resolve_path("sandbox_data") / "self_test_reports").as_posix(),
        env="SELF_TEST_REPORT_DIR",
        description="Directory storing self-test reports.",
    )
    synergy_weights_path: str = Field(
        (resolve_path("sandbox_data") / "synergy_weights.json").as_posix(),
        env="SYNERGY_WEIGHTS_PATH",
        description="Persisted synergy weight JSON file.",
    )
    synergy_weight_file: str = Field(
        (resolve_path("sandbox_data") / "synergy_weights.json").as_posix(),
        env="SYNERGY_WEIGHT_FILE",
        description="File storing persisted synergy weights between runs.",
    )
    default_synergy_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "roi": 1.0,
            "efficiency": 1.0,
            "resilience": 1.0,
            "antifragility": 1.0,
            "reliability": 1.0,
            "maintainability": 1.0,
            "throughput": 1.0,
        },
        env="DEFAULT_SYNERGY_WEIGHTS",
        description="Fallback synergy weights used when no persisted file is found.",
    )
    alignment_flags_path: str = Field(
        (resolve_path("sandbox_data") / "alignment_flags.jsonl").as_posix(),
        env="ALIGNMENT_FLAGS_PATH",
        description="Path for persisted alignment flag reports.",
    )
    module_synergy_graph_path: str = Field(
        (resolve_path("sandbox_data") / "module_synergy_graph.json").as_posix(),
        env="MODULE_SYNERGY_GRAPH_PATH",
        description="Synergy graph persistence path.",
    )
    relevancy_metrics_db_path: str = Field(
        (resolve_path("sandbox_data") / "relevancy_metrics.db").as_posix(),
        env="RELEVANCY_METRICS_DB_PATH",
        description="Database for relevancy metrics.",
    )
    synergy_learner: str = Field(
        "",
        env="SYNERGY_LEARNER",
        description="Override synergy learner backend.",
    )
    sandbox_auto_map: bool = Field(
        False,
        env="SANDBOX_AUTO_MAP",
        description="Automatically update module map after runs.",
    )
    sandbox_autodiscover_modules: bool = Field(
        False,
        env="SANDBOX_AUTODISCOVER_MODULES",
        description="Deprecated auto-discovery flag for module mapping.",
    )
    auto_train_synergy: bool = Field(
        False,
        env="AUTO_TRAIN_SYNERGY",
        description="Enable automatic periodic synergy training.",
    )
    auto_train_interval: float = Field(
        600.0,
        env="AUTO_TRAIN_INTERVAL",
        description="Interval in seconds between automatic synergy training runs.",
    )
    patch_score_backend_url: str | None = Field(
        None,
        env="PATCH_SCORE_BACKEND_URL",
        description="URL to patch score backend service.",
    )
    patch_score_backend: str | None = Field(
        None,
        env="PATCH_SCORE_BACKEND",
        description="Module path to patch score backend implementation.",
    )
    patch_retries: int = Field(
        3,
        env="SANDBOX_PATCH_RETRIES",
        description="Number of attempts made when generating a patch.",
    )
    patch_retry_delay: float = Field(
        0.1,
        env="SANDBOX_PATCH_RETRY_DELAY",
        description="Delay in seconds between patch generation attempts.",
    )
    diff_risk_threshold: float = Field(
        0.5,
        env="DIFF_RISK_THRESHOLD",
        description="Abort patches whose diff risk exceeds this score.",
    )
    flakiness_runs: int = Field(
        5,
        env="FLAKINESS_RUNS",
        description="Number of runs used to estimate test flakiness.",
    )
    weight_update_interval: float = Field(
        60.0,
        env="WEIGHT_UPDATE_INTERVAL",
        description="Minimum seconds between score weight recalculations.",
    )
    test_run_timeout: float = Field(
        300.0,
        env="TEST_RUN_TIMEOUT",
        description="Timeout in seconds for individual test runs.",
    )
    test_run_retries: int = Field(
        2,
        env="TEST_RUN_RETRIES",
        description="Retry attempts for failing test runs.",
    )
    orphan_reuse_threshold: float = Field(
        0.0,
        env="ORPHAN_REUSE_THRESHOLD",
        description="Minimum reuse score required for orphan modules.",
    )
    clean_orphans: bool = Field(
        False,
        env="SANDBOX_CLEAN_ORPHANS",
        description="Remove failing orphans from tracking file.",
    )
    disable_orphans: bool = Field(
        False,
        env="SANDBOX_DISABLE_ORPHANS",
        description="Disable orphan testing entirely.",
    )
    include_orphans: bool | None = Field(
        None,
        env="SANDBOX_INCLUDE_ORPHANS",
        description="Include orphan modules during testing when set.",
    )
    orphan_retry_attempts: int = Field(
        3,
        env="ORPHAN_RETRY_ATTEMPTS",
        description="Retry attempts for orphan integration hooks.",
    )
    orphan_retry_delay: float = Field(
        0.1,
        env="ORPHAN_RETRY_DELAY",
        description="Delay between retries for orphan integration hooks.",
    )
    exclude_dirs: str | None = Field(
        None,
        env="SANDBOX_EXCLUDE_DIRS",
        description="Comma-separated directories to exclude during scans.",
    )
    exploration_strategy: str = Field("epsilon_greedy", env="EXPLORATION_STRATEGY")
    exploration_epsilon: float = Field(0.1, env="EXPLORATION_EPSILON")
    exploration_temperature: float = Field(1.0, env="EXPLORATION_TEMPERATURE")
    policy_alpha: float = Field(
        0.5,
        env="POLICY_ALPHA",
        description="Learning rate for self-improvement policy updates.",
    )
    policy_gamma: float = Field(
        0.9,
        env="POLICY_GAMMA",
        description="Discount factor for future rewards in policy updates.",
    )
    policy_epsilon: float = Field(
        0.1,
        env="POLICY_EPSILON",
        description="Exploration rate for epsilon-greedy policy strategies.",
    )
    policy_temperature: float = Field(
        1.0,
        env="POLICY_TEMPERATURE",
        description="Temperature parameter for softmax exploration policies.",
    )
    policy_exploration: str = Field(
        "epsilon_greedy",
        env="POLICY_EXPLORATION",
        description="Exploration strategy for policy selection (e.g., 'epsilon_greedy').",
    )
    default_module_timeout: float | None = Field(
        None,
        env="SANDBOX_TIMEOUT",
        description="Default per-module execution timeout in seconds.",
    )
    default_memory_limit: int | None = Field(
        None,
        env="SANDBOX_MEMORY_LIMIT",
        description="Default per-module RSS memory limit in bytes.",
    )
    auto_include_isolated: bool = Field(
        True,
        env="SANDBOX_AUTO_INCLUDE_ISOLATED",
        description=(
            "Automatically include isolated modules during orphan scans "
            "(enabled by default)."
        ),
    )
    recursive_orphan_scan: bool = Field(
        True,
        env="SANDBOX_RECURSIVE_ORPHANS",
        description="Recurse through orphan dependencies when scanning (enabled by default).",
    )
    recursive_isolated: bool = Field(
        True,
        env="SANDBOX_RECURSIVE_ISOLATED",
        description="Recursively resolve local imports when auto-including isolated modules.",
    )
    max_recursion_depth: int | None = Field(
        None,
        env="SANDBOX_MAX_RECURSION_DEPTH",
        description="Maximum depth when resolving dependencies recursively (default unlimited).",
    )

    @field_validator("exploration_epsilon")
    def _validate_exploration_epsilon(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("exploration_epsilon must be between 0 and 1")
        return v

    @field_validator("exploration_temperature")
    def _validate_exploration_temperature(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("exploration_temperature must be positive")
        return v

    @field_validator("policy_alpha", "policy_gamma", "policy_epsilon")
    def _validate_policy_unit_range(cls, v: float, info: Any) -> float:
        if not 0 <= v <= 1:
            raise ValueError(f"{info.field_name} must be between 0 and 1")
        return v

    @field_validator("policy_temperature")
    def _validate_policy_temperature(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("policy_temperature must be positive")
        return v

    @field_validator("meta_entropy_threshold")
    def _validate_meta_entropy_threshold(cls, v: float | None) -> float | None:
        if v is not None and not 0 <= v <= 1:
            raise ValueError("meta_entropy_threshold must be between 0 and 1")
        return v

    @field_validator("overfitting_entropy_threshold")
    def _validate_overfitting_entropy_threshold(cls, v: float) -> float:
        if v < 0:
            raise ValueError("overfitting_entropy_threshold must be non-negative")
        return v

    @field_validator("meta_entropy_weight")
    def _validate_meta_entropy_weight(cls, v: float) -> float:
        if v < 0:
            raise ValueError("meta_entropy_weight must be non-negative")
        return v

    @field_validator("meta_search_depth", "meta_beam_width")
    def _validate_meta_search_params(cls, v: int, info: Any) -> int:
        if v <= 0:
            raise ValueError(f"{info.field_name} must be a positive integer")
        return v

    @field_validator("meta_planning_interval", "meta_planning_period")
    def _validate_meta_intervals(cls, v: int, info: Any) -> int:
        if v <= 0:
            raise ValueError(f"{info.field_name} must be a positive integer")
        return v

    @field_validator(
        "meta_improvement_threshold",
        "meta_mutation_rate",
        "meta_roi_weight",
        "meta_domain_penalty",
        "orphan_reuse_threshold",
    )
    def _validate_non_negative(cls, v: float, info: Any) -> float:
        if v < 0:
            raise ValueError(f"{info.field_name} must be non-negative")
        return v

    @field_validator("auto_train_interval")
    def _validate_auto_train_interval(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("auto_train_interval must be positive")
        return v

    @field_validator("flakiness_runs", "test_run_retries")
    def _validate_positive_int(cls, v: int, info: Any) -> int:
        if v <= 0:
            raise ValueError(f"{info.field_name} must be a positive integer")
        return v

    @field_validator(
        "adaptive_roi_retrain_interval",
        "adaptive_roi_train_interval",
        "backup_rotation_count",
        "checkpoint_retention",
    )
    def _validate_positive_training(cls, v: int, info: Any) -> int:
        if v <= 0:
            raise ValueError(f"{info.field_name} must be a positive integer")
        return v

    @field_validator(
        "weight_update_interval",
        "test_run_timeout",
        "side_effect_dev_multiplier",
        "synergy_dev_multiplier",
        "roi_threshold_k",
        "synergy_threshold_k",
    )
    def _validate_positive_float(cls, v: float, info: Any) -> float:
        if v <= 0:
            raise ValueError(f"{info.field_name} must be positive")
        return v

    test_redundant_modules: bool = Field(
        True,
        env="SANDBOX_TEST_REDUNDANT",
        description="Integrate modules classified as redundant after validation.",
    )
    side_effect_threshold: int = Field(10, env="SANDBOX_SIDE_EFFECT_THRESHOLD")
    side_effect_dev_multiplier: float = Field(
        1.0, env="SANDBOX_SIDE_EFFECT_DEV_MULTIPLIER"
    )
    synergy_dev_multiplier: float = Field(
        1.0, env="SANDBOX_SYNERGY_DEV_MULTIPLIER"
    )
    auto_patch_high_risk: bool = Field(
        True,
        env="AUTO_PATCH_HIGH_RISK",
        description="Enable automatic patching for high-risk modules.",
    )
    adaptive_roi_prioritization: bool = Field(
        True,
        env="ADAPTIVE_ROI_PRIORITIZATION",
        description="Prioritize improvements using AdaptiveROI classifications.",
    )
    self_coding_interval: int = Field(
        300,
        env="SELF_CODING_INTERVAL",
        description="Seconds between self-coding checks.",
    )
    self_coding_roi_drop: float = Field(
        -0.1,
        env="SELF_CODING_ROI_DROP",
        description="ROI drop triggering self-coding.",
    )
    self_coding_error_increase: float = Field(
        1.0,
        env="SELF_CODING_ERROR_INCREASE",
        description="Error increase triggering self-coding.",
    )
    self_coding_patch_success_drop: float = Field(
        -0.2,
        env="SELF_CODING_PATCH_SUCCESS_DROP",
        description="Patch success rate drop triggering self-coding.",
    )
    self_coding_test_command: list[str] | None = Field(
        None,
        env="SELF_CODING_TEST_COMMAND",
        description="Default test command used during patch approval.",
    )
    bot_thresholds: dict[str, BotThresholds] = Field(
        default_factory=dict,
        env="BOT_THRESHOLDS",
        description="Per-bot ROI drop and error thresholds.",
    )
    roi_growth_weighting: bool = Field(
        True,
        env="ROI_GROWTH_WEIGHTING",
        description="Weight rewards by predicted ROI growth categories.",
    )
    growth_multiplier_exponential: float = Field(
        3.0,
        env="GROWTH_MULTIPLIER_EXPONENTIAL",
        description="Multiplier applied to exponential growth predictions.",
    )
    growth_multiplier_linear: float = Field(
        2.0,
        env="GROWTH_MULTIPLIER_LINEAR",
        description="Multiplier applied to linear growth predictions.",
    )
    growth_multiplier_marginal: float = Field(
        1.0,
        env="GROWTH_MULTIPLIER_MARGINAL",
        description="Multiplier applied to marginal growth predictions.",
    )
    auto_dashboard_port: int | None = Field(None, env="AUTO_DASHBOARD_PORT")
    use_memory: bool = Field(
        True,
        env="SANDBOX_USE_MEMORY",
        description="Enable GPT memory integration during sandbox runs.",
    )
    use_module_synergy: bool = Field(
        False,
        env="SANDBOX_USE_MODULE_SYNERGY",
        description="Enable module synergy suggestions during workflow simulation and replacement.",
    )
    enable_truth_calibration: bool = Field(
        True,
        env="ENABLE_TRUTH_CALIBRATION",
        description="Enable TruthAdapter calibration of ROI metrics.",
    )
    psi_threshold: float | None = Field(
        None,
        env="PSI_THRESHOLD",
        description="Population Stability Index threshold for feature drift.",
    )
    ks_threshold: float | None = Field(
        None,
        env="KS_THRESHOLD",
        description="Kolmogorov–Smirnov statistic threshold for feature drift.",
    )
    roi_threshold: float | None = Field(None, env="ROI_THRESHOLD")
    synergy_threshold: float | None = Field(None, env="SYNERGY_THRESHOLD")
    roi_threshold_k: float = Field(1.0, env="ROI_THRESHOLD_K")
    synergy_threshold_k: float = Field(1.0, env="SYNERGY_THRESHOLD_K")
    roi_confidence: float | None = Field(
        None,
        env="ROI_CONFIDENCE",
        description="t-test confidence when flagging modules; defaults to 0.95",
    )
    synergy_confidence: float | None = Field(
        None,
        env="SYNERGY_CONFIDENCE",
        description="confidence level for synergy convergence checks; defaults to 0.95",
    )
    auto_merge_roi_threshold: float = Field(
        0.0, env="AUTO_MERGE_ROI_THRESHOLD"
    )
    auto_merge_coverage_threshold: float = Field(
        1.0, env="AUTO_MERGE_COVERAGE_THRESHOLD"
    )
    ac_actor_lr: float = Field(0.01, env="AC_ACTOR_LR")
    ac_critic_lr: float = Field(0.02, env="AC_CRITIC_LR")
    ac_gamma: float = Field(0.95, env="AC_GAMMA")
    ac_epsilon: float = Field(0.1, env="AC_EPSILON")
    ac_epsilon_decay: float = Field(0.99, env="AC_EPSILON_DECAY")
    ac_buffer_size: int = Field(100, env="AC_BUFFER_SIZE")
    ac_batch_size: int = Field(32, env="AC_BATCH_SIZE")
    ac_checkpoint_path: str = Field(
        "actor_critic_state.json", env="AC_CHECKPOINT_PATH"
    )
    ac_normalize_states: bool = Field(True, env="AC_NORMALIZE_STATES")
    ac_reward_scale: float = Field(1.0, env="AC_REWARD_SCALE")
    ac_eval_interval: int = Field(100, env="AC_EVAL_INTERVAL")
    ac_checkpoint_interval: int = Field(100, env="AC_CHECKPOINT_INTERVAL")
    workflow_merge_similarity: float = Field(
        0.9,
        env="WORKFLOW_MERGE_SIMILARITY",
        description="Minimum average efficiency/modularity required to merge workflows.",
    )
    workflow_merge_entropy_delta: float = Field(
        0.1,
        env="WORKFLOW_MERGE_ENTROPY_DELTA",
        description="Maximum entropy difference allowed when merging workflows.",
    )
    duplicate_similarity: float = Field(
        0.95,
        env="WORKFLOW_DUPLICATE_SIMILARITY",
        description="Minimum cosine similarity to treat workflows as duplicates.",
    )
    duplicate_entropy: float = Field(
        0.05,
        env="WORKFLOW_DUPLICATE_ENTROPY",
        description="Maximum entropy delta allowed when deduplicating workflows.",
    )
    best_practice_match_threshold: float = Field(
        0.9,
        env="WORKFLOW_BEST_PRACTICE_THRESHOLD",
        description=(
            "Similarity threshold against best-practice sequences required to "
            "treat a workflow as conforming to best practices."
        ),
    )
    borderline_raroi_threshold: float = Field(
        0.0,
        env="BORDERLINE_RAROI_THRESHOLD",
        description=(
            "RAROI below this value queues workflows in the borderline bucket."
        ),
    )
    borderline_confidence_threshold: float = Field(
        0.0,
        env="BORDERLINE_CONFIDENCE_THRESHOLD",
        description=(
            "Confidence below this value queues workflows in the borderline bucket."
        ),
    )
    micropilot_mode: str = Field(
        "auto",
        env="MICROPILOT_MODE",
        description=(
            "Borderline bucket handling: 'auto' runs micro-pilots immediately,"
            " 'queue' only enqueues, 'off' disables."
        ),
    )
    entropy_threshold: float | None = Field(None, env="ENTROPY_THRESHOLD")
    entropy_plateau_threshold: float | None = Field(
        None, env="ENTROPY_PLATEAU_THRESHOLD"
    )
    entropy_plateau_consecutive: int | None = Field(
        None, env="ENTROPY_PLATEAU_CONSECUTIVE"
    )
    entropy_ceiling_threshold: float | None = Field(
        None,
        env="ENTROPY_CEILING_THRESHOLD",
        description="ROI gain per entropy delta threshold used for module retirement decisions.",
    )
    entropy_ceiling_consecutive: int | None = Field(
        None,
        env="ENTROPY_CEILING_CONSECUTIVE",
        description=(
            "Number of consecutive cycles below the ceiling threshold before "
            "flagging a module."
        ),
    )
    entropy_window: int = Field(5, env="ENTROPY_WINDOW")
    entropy_weight: float = Field(0.1, env="ENTROPY_WEIGHT")
    roi_weight: float = Field(1.0, env="ROI_WEIGHT")
    momentum_weight: float = Field(1.0, env="MOMENTUM_WEIGHT")
    pass_rate_weight: float = Field(1.0, env="PASS_RATE_WEIGHT")
    entropy_weight_scale: float = Field(
        0.0,
        env="ENTROPY_WEIGHT_SCALE",
        description=(
            "Multiplier applied to the entropy standard deviation when "
            "adapting the entropy weight."
        ),
    )
    momentum_weight_scale: float = Field(
        0.0,
        env="MOMENTUM_WEIGHT_SCALE",
        description=(
            "Multiplier applied to the recent success ratio when "
            "adapting the momentum weight."
        ),
    )
    momentum_stagnation_dev_multiplier: float = Field(
        1.0,
        env="MOMENTUM_STAGNATION_DEV_MULTIPLIER",
        description=(
            "Multiplier for the momentum standard deviation when "
            "detecting stagnation."
        ),
    )
    min_integration_roi: float = Field(
        0.0,
        env="MIN_INTEGRATION_ROI",
        description="Minimum ROI increase required for module auto-integration.",
    )
    synergy_threshold_window: int | None = Field(None, env="SYNERGY_THRESHOLD_WINDOW")
    synergy_threshold_weight: float | None = Field(None, env="SYNERGY_THRESHOLD_WEIGHT")
    synergy_ma_window: int | None = Field(None, env="SYNERGY_MA_WINDOW")
    synergy_stationarity_confidence: float | None = Field(
        None,
        env="SYNERGY_STATIONARITY_CONFIDENCE",
        description="confidence for stationarity tests; defaults to 0.95",
    )
    synergy_std_threshold: float | None = Field(None, env="SYNERGY_STD_THRESHOLD")
    synergy_variance_confidence: float | None = Field(
        None,
        env="SYNERGY_VARIANCE_CONFIDENCE",
        description="confidence for variance tests; defaults to 0.95",
    )

    @field_validator(
        "roi_weight",
        "entropy_weight",
        "momentum_weight",
        "pass_rate_weight",
        "entropy_weight_scale",
        "momentum_weight_scale",
    )
    def _validate_delta_weights(cls, v: float, info: Any) -> float:
        if v < 0:
            raise ValueError(f"{info.field_name} must be non-negative")
        return v

    @field_validator("momentum_stagnation_dev_multiplier")
    def _momentum_stagnation_dev_multiplier_positive(
        cls, v: float
    ) -> float:
        if v <= 0:
            raise ValueError("momentum_stagnation_dev_multiplier must be positive")
        return v

    relevancy_threshold: int = Field(
        20,
        env="RELEVANCY_THRESHOLD",
        description="Minimum usage count before a module is considered relevant.",
    )
    relevancy_window_days: int = Field(
        30,
        env="RELEVANCY_WINDOW_DAYS",
        description="Days of history to consider when evaluating relevancy.",
    )
    relevancy_whitelist: list[str] = Field(
        default_factory=list,
        env="RELEVANCY_WHITELIST",
        description="Modules exempt from relevancy radar checks.",
    )

    enable_relevancy_radar: bool = Field(
        True,
        env="ENABLE_RELEVANCY_RADAR",
        description="Enable the relevancy radar during sandbox runs.",
    )
    relevancy_radar_interval: int = Field(
        3600,
        env="RELEVANCY_RADAR_INTERVAL",
        description="Interval in seconds between relevancy radar scans.",
    )
    relevancy_radar_min_calls: int = Field(
        0,
        env="RELEVANCY_RADAR_MIN_CALLS",
        description="Minimum invocation count considered when analysing module relevance.",
    )
    relevancy_radar_compress_ratio: float = Field(
        0.01,
        env="RELEVANCY_RADAR_COMPRESS_RATIO",
        description="Call and time ratio below which modules are flagged for compression.",
    )
    relevancy_radar_replace_ratio: float = Field(
        0.05,
        env="RELEVANCY_RADAR_REPLACE_RATIO",
        description="Call and time ratio below which modules are suggested for replacement.",
    )
    usage_queue_maxsize: int = Field(
        256,
        env="USAGE_QUEUE_MAXSIZE",
        description="Maximum number of module usage events queued for relevancy tracking.",
    )
    relevancy_metrics_retention_days: int | None = Field(
        None,
        env="RELEVANCY_METRICS_RETENTION_DAYS",
        description=(
            "Days to retain relevancy metrics. Older records are purged during "
            "on-demand radar scans when set."
        ),
    )
    auto_process_relevancy_flags: bool = Field(
        True,
        env="AUTO_PROCESS_RELEVANCY_FLAGS",
        description=(
            "Automatically process relevancy flags with ModuleRetirementService."
        ),
    )
    relevancy_deviation_multiplier: float = Field(
        1.0,
        env="RELEVANCY_DEVIATION_MULTIPLIER",
        description="Multiplier for std deviation when deriving relevancy thresholds.",
    )
    relevancy_history_min_length: int = Field(
        5,
        env="RELEVANCY_HISTORY_MIN_LENGTH",
        description="Minimum relevancy history length before auto-processing flags.",
    )

    synergy_weight_roi: float = Field(1.0, env="SYNERGY_WEIGHT_ROI")
    synergy_weight_efficiency: float = Field(1.0, env="SYNERGY_WEIGHT_EFFICIENCY")
    synergy_weight_resilience: float = Field(1.0, env="SYNERGY_WEIGHT_RESILIENCE")
    synergy_weight_antifragility: float = Field(1.0, env="SYNERGY_WEIGHT_ANTIFRAGILITY")
    synergy_weight_reliability: float = Field(1.0, env="SYNERGY_WEIGHT_RELIABILITY")
    synergy_weight_maintainability: float = Field(
        1.0, env="SYNERGY_WEIGHT_MAINTAINABILITY"
    )
    synergy_weight_throughput: float = Field(1.0, env="SYNERGY_WEIGHT_THROUGHPUT")
    roi_ema_alpha: float = Field(0.1, env="ROI_EMA_ALPHA")
    roi_compounding_weight: float = Field(1.0, env="ROI_COMPOUNDING_WEIGHT")
    roi_baseline_window: int = Field(5, env="ROI_BASELINE_WINDOW")
    roi_momentum_window: int = Field(5, env="ROI_MOMENTUM_WINDOW")
    roi_momentum_dev_multiplier: float = Field(1.0, env="ROI_MOMENTUM_DEV_MULTIPLIER")
    roi_stagnation_cycles: int = Field(3, env="ROI_STAGNATION_CYCLES")
    roi_deviation_tolerance: float = Field(0.05, env="ROI_DEVIATION_TOLERANCE")
    roi_stagnation_threshold: float = Field(0.01, env="ROI_STAGNATION_THRESHOLD")
    enable_snapshot_tracker: bool = Field(
        True,
        env="ENABLE_SNAPSHOT_TRACKER",
        description="Enable tracking of self-improvement snapshots and deltas.",
    )
    roi_drop_threshold: float = Field(
        0.0,
        env="ROI_DROP_THRESHOLD",
        description="ROI delta at or below this value flags a regression.",
    )
    entropy_regression_threshold: float = Field(
        0.0,
        env="ENTROPY_REGRESSION_THRESHOLD",
        description="Entropy delta at or below this value flags a regression.",
    )
    roi_penalty_threshold: float = Field(
        0.0,
        env="ROI_PENALTY_THRESHOLD",
        description="ROI delta at or below this value applies prompt penalties.",
    )
    entropy_penalty_threshold: float = Field(
        0.0,
        env="ENTROPY_PENALTY_THRESHOLD",
        description="Entropy delta at or above this value applies prompt penalties.",
    )
    sandbox_score_db: str = Field(
        (resolve_path(".") / "score_history.db").as_posix(),
        env="SANDBOX_SCORE_DB",
    )
    snapshot_dir: str = Field(
        (resolve_path("sandbox_data") / "snapshots").as_posix(),
        env="SNAPSHOT_DIR",
        description="Directory where state snapshots are stored.",
    )
    snapshot_diff_dir: str = Field(
        (resolve_path("sandbox_data") / "diffs").as_posix(),
        env="SNAPSHOT_DIFF_DIR",
        description="Directory where snapshot diffs are written.",
    )
    checkpoint_dir: str = Field(
        (resolve_path("sandbox_data") / "checkpoints").as_posix(),
        env="CHECKPOINT_DIR",
        description="Directory where state checkpoints are saved.",
    )
    checkpoint_retention: int = Field(
        5,
        env="CHECKPOINT_RETENTION",
        description="Number of checkpoint directories to retain.",
    )
    synergy_weights_lr: float = Field(
        0.1,
        env="SYNERGY_WEIGHTS_LR",
        description="Learning rate for synergy weight learner.",
    )
    synergy_train_interval: int = Field(
        10,
        env="SYNERGY_TRAIN_INTERVAL",
        description="Steps between learner optimisation updates.",
    )
    synergy_replay_size: int = Field(
        100,
        env="SYNERGY_REPLAY_SIZE",
        description="Length of replay buffer for RL strategies.",
    )
    synergy_batch_size: int = Field(
        32,
        env="SYNERGY_BATCH_SIZE",
        description="Mini-batch size for RL learner updates.",
    )
    synergy_gamma: float = Field(
        0.99,
        env="SYNERGY_GAMMA",
        description="Discount factor for RL learner updates.",
    )
    synergy_noise: float = Field(
        0.1,
        env="SYNERGY_NOISE",
        description="Exploration noise added to learner actions.",
    )
    synergy_hidden_size: int = Field(
        32,
        env="SYNERGY_HIDDEN_SIZE",
        description="Hidden units per layer for torch models.",
    )
    synergy_layers: int = Field(
        1,
        env="SYNERGY_LAYERS",
        description="Number of hidden layers for torch models.",
    )
    synergy_optimizer: str = Field(
        "adam",
        env="SYNERGY_OPTIMIZER",
        description="Optimizer to use for torch strategies.",
    )
    synergy_checkpoint_interval: int = Field(
        50,
        env="SYNERGY_CHECKPOINT_INTERVAL",
        description="Updates between on-disk checkpoints.",
    )
    synergy_strategy: str = Field(
        "dqn",
        env="SYNERGY_STRATEGY",
        description="DQN variant for DQNSynergyLearner.",
    )
    synergy_target_sync: int = Field(
        10,
        env="SYNERGY_TARGET_SYNC",
        description="Steps between target network synchronisation.",
    )
    synergy_python_fallback: bool = Field(
        True,
        env="SYNERGY_PYTHON_FALLBACK",
        description="Allow slow pure-Python learner when torch missing.",
    )
    synergy_python_max_replay: int = Field(
        1000,
        env="SYNERGY_PYTHON_MAX_REPLAY",
        description="Replay size limit for Python fallback before requiring torch.",
    )
    adaptive_roi_retrain_interval: int = Field(
        20,
        env="ADAPTIVE_ROI_RETRAIN_INTERVAL",
        description="Cycles between adaptive ROI model retraining.",
    )
    adaptive_roi_train_interval: int = Field(
        3600,
        env="ADAPTIVE_ROI_TRAIN_INTERVAL",
        description="Seconds between scheduled adaptive ROI predictor training.",
    )
    backup_rotation_count: int = Field(
        3,
        env="SELF_IMPROVEMENT_BACKUP_COUNT",
        description="Number of rotated backups to keep for self-improvement data.",
    )
    scenario_metric_thresholds: dict[str, float] = Field(
        default_factory=lambda: {
            "schema_mismatch_rate": 0.1,
            "upstream_failure_rate": 0.1,
        },
        env="SCENARIO_METRIC_THRESHOLDS",
        description=(
            "Thresholds for scenario-specific metrics returned by "
            "_scenario_specific_metrics."
        ),
    )
    fail_on_missing_scenarios: bool = Field(
        False,
        env="SANDBOX_FAIL_ON_MISSING_SCENARIOS",
        description="Raise an error when canonical scenarios are missing coverage.",
    )

    # Scenario preset defaults
    preset_concurrency_multiplier: int = Field(
        4,
        env="SANDBOX_PRESET_CONCURRENCY_MULTIPLIER",
        description="Multiplier applied to base concurrency in spike scenarios.",
    )
    preset_concurrency_level: int = Field(
        8,
        env="SANDBOX_PRESET_CONCURRENCY_LEVEL",
        description="Base concurrency level used in spike scenarios.",
    )
    preset_hostile_stub_strategy: str = Field(
        "hostile",
        env="SANDBOX_PRESET_HOSTILE_STUB_STRATEGY",
        description="Stub strategy employed for hostile input scenarios.",
    )
    preset_hostile_input: bool = Field(
        True,
        env="SANDBOX_PRESET_HOSTILE_INPUT",
        description="Toggle to inject adversarial inputs in hostile scenarios.",
    )
    preset_schema_stub_strategy: str = Field(
        "legacy_schema",
        env="SANDBOX_PRESET_SCHEMA_STUB_STRATEGY",
        description="Stub strategy used to emulate legacy schemas.",
    )
    preset_schema_mismatches: int = Field(
        5,
        env="SANDBOX_PRESET_SCHEMA_MISMATCHES",
        description="Number of schema mismatches injected in schema drift scenarios.",
    )
    preset_schema_checks: int = Field(
        100,
        env="SANDBOX_PRESET_SCHEMA_CHECKS",
        description="Total schema checks performed in schema drift scenarios.",
    )
    preset_upstream_failures: int = Field(
        1,
        env="SANDBOX_PRESET_UPSTREAM_FAILURES",
        description="Number of simulated upstream failures in flaky scenarios.",
    )
    preset_upstream_requests: int = Field(
        20,
        env="SANDBOX_PRESET_UPSTREAM_REQUESTS",
        description="Total upstream requests made in flaky scenarios.",
    )
    preset_flaky_stub_strategy: str = Field(
        "flaky_upstream",
        env="SANDBOX_PRESET_FLAKY_STUB_STRATEGY",
        description="Stub strategy used to emulate flaky upstream dependencies.",
    )
    preset_api_latency_ms: int = Field(
        500,
        env="SANDBOX_PRESET_API_LATENCY_MS",
        description="API latency in milliseconds for flaky upstream scenarios.",
    )
    preset_network_latency_ms: int = Field(
        500,
        env="SANDBOX_PRESET_NETWORK_LATENCY_MS",
        description="Network latency in milliseconds for latency scenarios.",
    )
    preset_cpu_limit: float = Field(
        0.5,
        env="SANDBOX_PRESET_CPU_LIMIT",
        description="CPU limit used for resource strain scenarios.",
    )
    preset_disk_limit: str = Field(
        "512mb",
        env="SANDBOX_PRESET_DISK_LIMIT",
        description="Disk limit used for resource strain scenarios.",
    )

    alignment_rules: AlignmentRules = Field(
        default_factory=AlignmentRules,
        description="Thresholds for human-alignment checks used by flaggers.",
    )

    enable_alignment_flagger: bool = Field(
        True,
        env="ENABLE_ALIGNMENT_FLAGGER",
        description=(
            "Run the human-alignment flagger after each commit. Enabled by default to "
            "surface potential safety regressions early."
        ),
    )
    alignment_warning_threshold: float = Field(
        0.5,
        env="ALIGNMENT_WARNING_THRESHOLD",
        description=(
            "Normalised issue severity at or above this value raises non-blocking "
            "warnings. The default of 0.5 balances sensitivity with noise."
        ),
    )
    alignment_failure_threshold: float = Field(
        0.9,
        env="ALIGNMENT_FAILURE_THRESHOLD",
        description=(
            "Normalised issue severity at or above this value is considered "
            "severe. A high default of 0.9 avoids false positives while still "
            "flagging critical issues."
        ),
    )
    improvement_warning_threshold: float = Field(
        0.5,
        env="IMPROVEMENT_WARNING_THRESHOLD",
        description=(
            "Normalised severity at or above this value triggers logging of "
            "self-improvement warnings."
        ),
    )
    improvement_failure_threshold: float = Field(
        0.9,
        env="IMPROVEMENT_FAILURE_THRESHOLD",
        description=(
            "Normalised severity at or above this value marks self-improvement "
            "issues as failures."
        ),
    )
    alignment_baseline_metrics_path: Path = Field(
        default_factory=lambda: resolve_path("sandbox_metrics.yaml"),
        env="ALIGNMENT_BASELINE_METRICS_PATH",
        description=(
            "Path to baseline metrics file for maintainability comparisons. By "
            "default this points to the repository's sandbox_metrics.yaml snapshot."
        ),
    )

    metrics_skip_dirs: list[str] = Field(
        default_factory=lambda: [
            ".git",
            "bin",
            "build",
            "dist",
            "node_modules",
            "site-packages",
            "venv",
            ".venv",
            "vendor",
            "third_party",
            "__pycache__",
        ],
        env="METRICS_SKIP_DIRS",
        description="Directories to skip when collecting code metrics.",
    )

    snapshot_metrics: list[str] = Field(
        default_factory=lambda: [
            "roi",
            "sandbox_score",
            "entropy",
            "call_graph_complexity",
            "token_diversity",
        ],
        env="SNAPSHOT_METRICS",
        description="Metrics captured and tracked for each self-improvement snapshot.",
    )

    # self test integration scoring knobs
    integration_score_threshold: float = Field(
        0.0,
        env="INTEGRATION_SCORE_THRESHOLD",
        description="Minimum score required for module auto-integration.",
    )
    integration_weight_coverage: float = Field(
        1.0,
        env="INTEGRATION_WEIGHT_COVERAGE",
        description="Weight applied to coverage percentage when scoring modules.",
    )
    integration_weight_runtime: float = Field(
        1.0,
        env="INTEGRATION_WEIGHT_RUNTIME",
        description="Weight applied to runtime when scoring modules (higher reduces score).",
    )
    integration_weight_failures: float = Field(
        1.0,
        env="INTEGRATION_WEIGHT_FAILURES",
        description="Penalty weight for each failure category detected.",
    )

    # self modification detector configuration
    self_mod_interval_seconds: int = Field(
        10,
        env="SELF_MOD_INTERVAL_SECONDS",
        description="Seconds between integrity checks for self-modification detection.",
    )
    self_mod_reference_path: str = Field(
        (resolve_path(".") / "immutable_reference.json").as_posix(),
        env="SELF_MOD_REFERENCE_PATH",
        description="Path to reference hash snapshot.",
    )
    self_mod_reference_url: str | None = Field(
        None,
        env="SELF_MOD_REFERENCE_URL",
        description="Optional URL providing reference hashes.",
    )
    self_mod_lockdown_flag_path: str = Field(
        (resolve_path(".") / "lockdown.flag").as_posix(),
        env="SELF_MOD_LOCKDOWN_FLAG_PATH",
        description="Location of lockdown flag written on tampering.",
    )

    # self debugger scoring configuration
    baseline_window: int = Field(
        5,
        env="BASELINE_WINDOW",
        description="Number of recent scores used for moving average baseline.",
    )
    stagnation_iters: int = Field(
        10,
        env="STAGNATION_ITERS",
        description=(
            "Iterations with no improvement before the baseline resets to the "
            "current average."
        ),
    )
    delta_margin: float = Field(
        0.0,
        env="DELTA_MARGIN",
        description="Minimum positive delta over baseline required for patch acceptance.",
    )
    score_weights: tuple[float, float, float, float, float, float] = Field(
        (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        env="SCORE_WEIGHTS",
        description="Weights for coverage, errors, ROI, complexity, synergy ROI and efficiency.",
    )

    @field_validator("baseline_window", "stagnation_iters")
    def _check_positive_int(cls, v: int, info: Any) -> int:
        if v <= 0:
            raise ValueError(f"{info.field_name} must be a positive integer")
        return v

    @field_validator("delta_margin")
    def _check_delta_margin(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("delta_margin must be between 0 and 1")
        return v

    @field_validator(
        "score_weights",
        **({"mode": "before"} if PYDANTIC_V2 else {"pre": True}),
    )
    def _parse_score_weights(
        cls, v: Any
    ) -> tuple[float, float, float, float, float, float]:
        if isinstance(v, str):
            try:
                v = (
                    json.loads(v)
                    if v.strip().startswith("[")
                    else [float(x) for x in v.split(",")]
                )
            except Exception as e:  # pragma: no cover - defensive
                raise ValueError("score_weights must be a list of floats") from e
        return tuple(v)

    @field_validator("score_weights")
    def _check_score_weights(
        cls, v: tuple[float, float, float, float, float, float]
    ) -> tuple[float, float, float, float, float, float]:
        if len(v) != 6:
            raise ValueError("score_weights must contain six values")
        if any(w < 0 for w in v):
            raise ValueError("score_weights values must be non-negative")
        return v

    @field_validator("sandbox_data_dir", "self_test_report_dir")
    def _ensure_dirs(cls, v: str) -> str:
        try:
            path = resolve_path(v)
        except FileNotFoundError:
            path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path.as_posix()

    @field_validator(
        "self_test_lock_file",
        "synergy_weights_path",
        "alignment_flags_path",
        "module_synergy_graph_path",
        "relevancy_metrics_db_path",
        "self_mod_reference_path",
        "self_mod_lockdown_flag_path",
    )
    def _ensure_parent_dirs(cls, v: str) -> str:
        try:
            path = resolve_path(v)
        except FileNotFoundError:
            path = Path(v)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.as_posix()

    # Grouped settings
    roi: ROISettings = Field(default_factory=ROISettings, exclude=True)
    synergy: SynergySettings = Field(default_factory=SynergySettings, exclude=True)
    alignment: AlignmentSettings = Field(
        default_factory=AlignmentSettings, exclude=True
    )
    auto_merge: AutoMergeSettings = Field(
        default_factory=AutoMergeSettings, exclude=True
    )
    actor_critic: ActorCriticSettings = Field(
        default_factory=ActorCriticSettings, exclude=True
    )
    policy: PolicySettings = Field(
        default_factory=PolicySettings, exclude=True
    )

    def __init__(self, **data: Any) -> None:  # pragma: no cover - simple wiring
        super().__init__(**data)
        self.roi = ROISettings(
            threshold=self.roi_threshold,
            confidence=self.roi_confidence,
            ema_alpha=self.roi_ema_alpha,
            compounding_weight=self.roi_compounding_weight,
            entropy_window=self.entropy_window,
            entropy_weight=self.entropy_weight,
            baseline_window=self.roi_baseline_window,
            deviation_tolerance=self.roi_deviation_tolerance,
            stagnation_threshold=self.roi_stagnation_threshold,
            momentum_window=self.roi_momentum_window,
            stagnation_cycles=self.roi_stagnation_cycles,
            momentum_dev_multiplier=self.roi_momentum_dev_multiplier,
            min_integration_roi=self.min_integration_roi,
            entropy_threshold=self.entropy_threshold,
            entropy_plateau_threshold=self.entropy_plateau_threshold,
            entropy_plateau_consecutive=self.entropy_plateau_consecutive,
            entropy_ceiling_threshold=self.entropy_ceiling_threshold,
            entropy_ceiling_consecutive=self.entropy_ceiling_consecutive,
        )
        self.synergy = SynergySettings(
            threshold=self.synergy_threshold,
            confidence=self.synergy_confidence,
            threshold_window=self.synergy_threshold_window,
            threshold_weight=self.synergy_threshold_weight,
            ma_window=self.synergy_ma_window,
            stationarity_confidence=self.synergy_stationarity_confidence,
            std_threshold=self.synergy_std_threshold,
            variance_confidence=self.synergy_variance_confidence,
            weight_roi=self.synergy_weight_roi,
            weight_efficiency=self.synergy_weight_efficiency,
            weight_resilience=self.synergy_weight_resilience,
            weight_antifragility=self.synergy_weight_antifragility,
            weight_reliability=self.synergy_weight_reliability,
            weight_maintainability=self.synergy_weight_maintainability,
            weight_throughput=self.synergy_weight_throughput,
            weights_lr=self.synergy_weights_lr,
            train_interval=self.synergy_train_interval,
            replay_size=self.synergy_replay_size,
            batch_size=self.synergy_batch_size,
            gamma=self.synergy_gamma,
            noise=self.synergy_noise,
            hidden_size=self.synergy_hidden_size,
            layers=self.synergy_layers,
            optimizer=self.synergy_optimizer,
            checkpoint_interval=self.synergy_checkpoint_interval,
            strategy=self.synergy_strategy,
            target_sync=self.synergy_target_sync,
            python_fallback=self.synergy_python_fallback,
            python_max_replay=self.synergy_python_max_replay,
        )
        self.alignment = AlignmentSettings(
            rules=self.alignment_rules,
            enable_flagger=self.enable_alignment_flagger,
            warning_threshold=self.alignment_warning_threshold,
            failure_threshold=self.alignment_failure_threshold,
            improvement_warning_threshold=self.improvement_warning_threshold,
            improvement_failure_threshold=self.improvement_failure_threshold,
            baseline_metrics_path=self.alignment_baseline_metrics_path,
        )
        self.auto_merge = AutoMergeSettings(
            roi_threshold=self.auto_merge_roi_threshold,
            coverage_threshold=self.auto_merge_coverage_threshold,
        )
        self.actor_critic = ActorCriticSettings(
            actor_lr=self.ac_actor_lr,
            critic_lr=self.ac_critic_lr,
            gamma=self.ac_gamma,
            epsilon=self.ac_epsilon,
            epsilon_decay=self.ac_epsilon_decay,
            buffer_size=self.ac_buffer_size,
            batch_size=self.ac_batch_size,
            checkpoint_path=self.ac_checkpoint_path,
            normalize_states=self.ac_normalize_states,
            reward_scale=self.ac_reward_scale,
            eval_interval=self.ac_eval_interval,
            checkpoint_interval=self.ac_checkpoint_interval,
        )
        self.policy = PolicySettings(
            alpha=self.policy_alpha,
            gamma=self.policy_gamma,
            epsilon=self.policy_epsilon,
            temperature=self.policy_temperature,
            exploration=self.policy_exploration,
        )

    model_config = SettingsConfigDict(
        env_file=os.getenv("MENACE_ENV_FILE", ".env"),
        extra="ignore",
    )

    if not PYDANTIC_V2:

        class Config:  # pragma: no cover - fallback for pydantic<2
            env_file = os.getenv("MENACE_ENV_FILE", ".env")
            extra = "ignore"


def load_sandbox_settings(path: str | None = None) -> SandboxSettings:
    """Load :class:`SandboxSettings` from optional YAML/JSON file."""

    data: dict[str, Any] = {}
    if path:
        path = resolve_path(path)
        with open(path, "r", encoding="utf-8") as fh:
            if path.suffix in (".yml", ".yaml"):
                data = yaml.safe_load(fh) or {}
            elif path.suffix == ".json":
                data = json.load(fh)
            else:  # pragma: no cover - defensive
                raise ValueError(f"Unsupported config format: {path}")
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
]
