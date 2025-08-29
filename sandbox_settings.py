from __future__ import annotations

"""Pydantic settings for sandbox utilities."""

import os
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    PYDANTIC_V2 = True
except Exception:  # pragma: no cover - fallback for pydantic<2
    from pydantic import BaseSettings  # type: ignore
    PYDANTIC_V2 = False
    SettingsConfigDict = dict  # type: ignore[misc]
from pydantic import BaseModel, Field


class AlignmentRules(BaseModel):
    """Thresholds for human-alignment checks."""

    max_complexity_score: int = 10
    max_comment_density_drop: float = 0.1
    allowed_network_calls: int = 0
    comment_density_severity: int = 2
    network_call_severity: int = 3
    rule_modules: list[str] = Field(default_factory=list)


class SandboxSettings(BaseSettings):
    """Environment configuration for sandbox runners."""

    menace_mode: str = Field("test", env="MENACE_MODE")
    database_url: str = Field("", env="DATABASE_URL")
    menace_offline_install: bool = Field(False, env="MENACE_OFFLINE_INSTALL")
    menace_wheel_dir: str | None = Field(None, env="MENACE_WHEEL_DIR")
    roi_cycles: int | None = Field(None, env="ROI_CYCLES")
    synergy_cycles: int | None = Field(None, env="SYNERGY_CYCLES")
    save_synergy_history: bool | None = Field(None, env="SAVE_SYNERGY_HISTORY")
    menace_env_file: str = Field(".env", env="MENACE_ENV_FILE")
    sandbox_data_dir: str = Field("sandbox_data", env="SANDBOX_DATA_DIR")
    sandbox_env_presets: str | None = Field(None, env="SANDBOX_ENV_PRESETS")
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
    synergy_std_threshold: float | None = Field(
        None, env="SYNERGY_STD_THRESHOLD"
    )
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
    synergy_weight_maintainability: float = Field(1.0, env="SYNERGY_WEIGHT_MAINTAINABILITY")
    synergy_weight_throughput: float = Field(1.0, env="SYNERGY_WEIGHT_THROUGHPUT")
    roi_ema_alpha: float = Field(0.1, env="ROI_EMA_ALPHA")
    roi_compounding_weight: float = Field(1.0, env="ROI_COMPOUNDING_WEIGHT")
    sandbox_score_db: str = Field("score_history.db", env="SANDBOX_SCORE_DB")
    synergy_weights_lr: float = Field(0.1, env="SYNERGY_WEIGHTS_LR")
    synergy_train_interval: int = Field(10, env="SYNERGY_TRAIN_INTERVAL")
    synergy_replay_size: int = Field(100, env="SYNERGY_REPLAY_SIZE")
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

    model_config = SettingsConfigDict(
        env_file=os.getenv("MENACE_ENV_FILE", ".env"),
        extra="ignore",
    )

    if not PYDANTIC_V2:
        class Config:  # pragma: no cover - fallback for pydantic<2
            env_file = os.getenv("MENACE_ENV_FILE", ".env")
            extra = "ignore"


__all__ = ["SandboxSettings", "AlignmentRules"]
