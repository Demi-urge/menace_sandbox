from __future__ import annotations

"""Pydantic settings for sandbox utilities."""

import json
import os
from typing import Any
from pathlib import Path

import yaml

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
    from pydantic import validator as field_validator  # type: ignore


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
    entropy_threshold: float | None = None
    entropy_plateau_threshold: float | None = None
    entropy_plateau_consecutive: int | None = None
    entropy_ceiling_threshold: float | None = None
    entropy_ceiling_consecutive: int | None = None

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

    @field_validator("compounding_weight", "min_integration_roi")
    def _check_non_negative(cls, v: float, info: Any) -> float:
        if v < 0:
            raise ValueError(f"{info.field_name} must be non-negative")
        return v

    @field_validator("entropy_plateau_consecutive", "entropy_ceiling_consecutive")
    def _check_positive(cls, v: int | None, info: Any) -> int | None:
        if v is not None and v <= 0:
            raise ValueError(f"{info.field_name} must be a positive integer")
        return v


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
    hidden_size: int = 32
    layers: int = 1
    optimizer: str = "adam"
    checkpoint_interval: int = 50

    @field_validator(
        "threshold",
        "confidence",
        "threshold_weight",
        "stationarity_confidence",
        "std_threshold",
        "variance_confidence",
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
        "hidden_size",
        "layers",
        "checkpoint_interval",
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
    )
    def _synergy_non_negative(cls, v: float, info: Any) -> float:
        if v < 0:
            raise ValueError(f"{info.field_name} must be non-negative")
        return v


class AlignmentSettings(BaseModel):
    """Grouping of alignment-related settings."""

    rules: AlignmentRules = Field(default_factory=AlignmentRules)
    enable_flagger: bool = True
    warning_threshold: float = 0.5
    failure_threshold: float = 0.9
    improvement_warning_threshold: float = 0.5
    improvement_failure_threshold: float = 0.9
    baseline_metrics_path: str = "sandbox_metrics.yaml"

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


class SandboxSettings(BaseSettings):
    """Environment configuration for sandbox runners."""

    menace_mode: str = Field("test", env="MENACE_MODE")
    database_url: str = Field("", env="DATABASE_URL")
    menace_offline_install: bool = Field(False, env="MENACE_OFFLINE_INSTALL")
    menace_wheel_dir: str | None = Field(None, env="MENACE_WHEEL_DIR")
    menace_light_imports: bool = Field(False, env="MENACE_LIGHT_IMPORTS")
    roi_cycles: int | None = Field(None, env="ROI_CYCLES")
    synergy_cycles: int | None = Field(None, env="SYNERGY_CYCLES")
    save_synergy_history: bool | None = Field(None, env="SAVE_SYNERGY_HISTORY")
    menace_env_file: str = Field(".env", env="MENACE_ENV_FILE")
    sandbox_data_dir: str = Field("sandbox_data", env="SANDBOX_DATA_DIR")
    sandbox_env_presets: str | None = Field(None, env="SANDBOX_ENV_PRESETS")
    sandbox_repo_path: str = Field(
        ".",
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
    visual_agent_monitor_interval: float = Field(
        30.0, env="VISUAL_AGENT_MONITOR_INTERVAL"
    )
    local_knowledge_refresh_interval: float = Field(
        600.0, env="LOCAL_KNOWLEDGE_REFRESH_INTERVAL"
    )
    menace_local_db_path: str | None = Field(None, env="MENACE_LOCAL_DB_PATH")
    menace_shared_db_path: str = Field(
        "./shared/global.db", env="MENACE_SHARED_DB_PATH"
    )
    visual_agent_token: str = Field("", env="VISUAL_AGENT_TOKEN")
    visual_agent_token_rotate: str | None = Field(None, env="VISUAL_AGENT_TOKEN_ROTATE")
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
    retry_optional_dependencies: bool = Field(
        False,
        env="RETRY_OPTIONAL_DEPENDENCIES",
        description="Retry loading optional modules instead of using a stub.",
    )
    openai_api_key: str | None = Field(
        None, env="OPENAI_API_KEY", description="API key for OpenAI access."
    )
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
    stub_timeout: float = Field(
        10.0,
        env="SANDBOX_STUB_TIMEOUT",
        description="Timeout for stub generation in seconds.",
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
        "sandbox_data/input_stub_templates.json",
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
    meta_entropy_threshold: float | None = Field(
        0.2,
        env="META_ENTROPY_THRESHOLD",
        description=(
            "Maximum allowed workflow entropy when recording improvements."
            " Defaults to 0.2 when unset."
        ),
    )
    workflows_db: str = Field(
        "workflows.db",
        env="WORKFLOWS_DB",
        description="SQLite database storing workflow definitions.",
    )
    gpt_memory_db: str = Field(
        "gpt_memory.db",
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
        "sandbox_data/self_test.lock",
        env="SELF_TEST_LOCK_FILE",
        description="File used to serialise self-test runs.",
    )
    self_test_report_dir: str = Field(
        "sandbox_data/self_test_reports",
        env="SELF_TEST_REPORT_DIR",
        description="Directory storing self-test reports.",
    )
    synergy_weights_path: str = Field(
        "sandbox_data/synergy_weights.json",
        env="SYNERGY_WEIGHTS_PATH",
        description="Persisted synergy weight JSON file.",
    )
    synergy_weight_file: str = Field(
        "sandbox_data/synergy_weights.json",
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
        "sandbox_data/alignment_flags.jsonl",
        env="ALIGNMENT_FLAGS_PATH",
        description="Path for persisted alignment flag reports.",
    )
    module_synergy_graph_path: str = Field(
        "sandbox_data/module_synergy_graph.json",
        env="MODULE_SYNERGY_GRAPH_PATH",
        description="Synergy graph persistence path.",
    )
    relevancy_metrics_db_path: str = Field(
        "sandbox_data/relevancy_metrics.db",
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
    exclude_dirs: str | None = Field(
        None,
        env="SANDBOX_EXCLUDE_DIRS",
        description="Comma-separated directories to exclude during scans.",
    )
    exploration_strategy: str = Field("epsilon_greedy", env="EXPLORATION_STRATEGY")
    exploration_epsilon: float = Field(0.1, env="EXPLORATION_EPSILON")
    exploration_temperature: float = Field(1.0, env="EXPLORATION_TEMPERATURE")
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

    @field_validator("meta_entropy_threshold")
    def _validate_meta_entropy_threshold(cls, v: float | None) -> float | None:
        if v is not None and not 0 <= v <= 1:
            raise ValueError("meta_entropy_threshold must be between 0 and 1")
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
    )
    def _validate_positive_training(cls, v: int, info: Any) -> int:
        if v <= 0:
            raise ValueError(f"{info.field_name} must be a positive integer")
        return v

    @field_validator("weight_update_interval", "test_run_timeout")
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
    visual_agent_autostart: bool = Field(True, env="VISUAL_AGENT_AUTOSTART")
    visual_agent_urls: str = Field("http://127.0.0.1:8001", env="VISUAL_AGENT_URLS")
    va_prompt_template: str | None = Field(None, env="VA_PROMPT_TEMPLATE")
    va_prompt_prefix: str = Field("", env="VA_PROMPT_PREFIX")
    va_repo_layout_lines: int = Field(20, env="VA_REPO_LAYOUT_LINES")
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
    roi_confidence: float | None = Field(None, env="ROI_CONFIDENCE")
    synergy_confidence: float | None = Field(None, env="SYNERGY_CONFIDENCE")
    auto_merge_roi_threshold: float = Field(
        0.0, env="AUTO_MERGE_ROI_THRESHOLD"
    )
    auto_merge_coverage_threshold: float = Field(
        1.0, env="AUTO_MERGE_COVERAGE_THRESHOLD"
    )
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
    min_integration_roi: float = Field(
        0.0,
        env="MIN_INTEGRATION_ROI",
        description="Minimum ROI increase required for module auto-integration.",
    )
    synergy_threshold_window: int | None = Field(None, env="SYNERGY_THRESHOLD_WINDOW")
    synergy_threshold_weight: float | None = Field(None, env="SYNERGY_THRESHOLD_WEIGHT")
    synergy_ma_window: int | None = Field(None, env="SYNERGY_MA_WINDOW")
    synergy_stationarity_confidence: float | None = Field(
        None, env="SYNERGY_STATIONARITY_CONFIDENCE"
    )
    synergy_std_threshold: float | None = Field(None, env="SYNERGY_STD_THRESHOLD")
    synergy_variance_confidence: float | None = Field(
        None, env="SYNERGY_VARIANCE_CONFIDENCE"
    )

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
    sandbox_score_db: str = Field("score_history.db", env="SANDBOX_SCORE_DB")
    synergy_weights_lr: float = Field(0.1, env="SYNERGY_WEIGHTS_LR")
    synergy_train_interval: int = Field(10, env="SYNERGY_TRAIN_INTERVAL")
    synergy_replay_size: int = Field(100, env="SYNERGY_REPLAY_SIZE")
    synergy_hidden_size: int = Field(32, env="SYNERGY_HIDDEN_SIZE")
    synergy_layers: int = Field(1, env="SYNERGY_LAYERS")
    synergy_optimizer: str = Field("adam", env="SYNERGY_OPTIMIZER")
    synergy_checkpoint_interval: int = Field(50, env="SYNERGY_CHECKPOINT_INTERVAL")
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
    alignment_baseline_metrics_path: str = Field(
        "sandbox_metrics.yaml",
        env="ALIGNMENT_BASELINE_METRICS_PATH",
        description=(
            "Path to baseline metrics file for maintainability comparisons. By "
            "default this points to the repository's sandbox_metrics.yaml snapshot."
        ),
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
        "immutable_reference.json",
        env="SELF_MOD_REFERENCE_PATH",
        description="Path to reference hash snapshot.",
    )
    self_mod_reference_url: str | None = Field(
        None,
        env="SELF_MOD_REFERENCE_URL",
        description="Optional URL providing reference hashes.",
    )
    self_mod_lockdown_flag_path: str = Field(
        "lockdown.flag",
        env="SELF_MOD_LOCKDOWN_FLAG_PATH",
        description="Location of lockdown flag written on tampering.",
    )

    # self debugger scoring configuration
    score_threshold: float = Field(
        0.5,
        env="SCORE_THRESHOLD",
        description="Minimum composite score required for patch acceptance.",
    )
    score_weights: tuple[float, float, float, float, float, float] = Field(
        (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        env="SCORE_WEIGHTS",
        description="Weights for coverage, errors, ROI, complexity, synergy ROI and efficiency.",
    )

    @field_validator("score_threshold")
    def _check_score_threshold(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("score_threshold must be between 0 and 1")
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
        Path(v).mkdir(parents=True, exist_ok=True)
        return v

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
        Path(v).parent.mkdir(parents=True, exist_ok=True)
        return v

    # Grouped settings
    roi: ROISettings = Field(default_factory=ROISettings, exclude=True)
    synergy: SynergySettings = Field(default_factory=SynergySettings, exclude=True)
    alignment: AlignmentSettings = Field(
        default_factory=AlignmentSettings, exclude=True
    )
    auto_merge: AutoMergeSettings = Field(
        default_factory=AutoMergeSettings, exclude=True
    )

    def __init__(self, **data: Any) -> None:  # pragma: no cover - simple wiring
        super().__init__(**data)
        self.roi = ROISettings(
            threshold=self.roi_threshold,
            confidence=self.roi_confidence,
            ema_alpha=self.roi_ema_alpha,
            compounding_weight=self.roi_compounding_weight,
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
            hidden_size=self.synergy_hidden_size,
            layers=self.synergy_layers,
            optimizer=self.synergy_optimizer,
            checkpoint_interval=self.synergy_checkpoint_interval,
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
        with open(path, "r", encoding="utf-8") as fh:
            if path.endswith((".yml", ".yaml")):
                data = yaml.safe_load(fh) or {}
            elif path.endswith(".json"):
                data = json.load(fh)
            else:  # pragma: no cover - defensive
                raise ValueError(f"Unsupported config format: {path}")
    return SandboxSettings(**data)


__all__ = [
    "SandboxSettings",
    "AlignmentRules",
    "ROISettings",
    "SynergySettings",
    "AlignmentSettings",
    "load_sandbox_settings",
]
