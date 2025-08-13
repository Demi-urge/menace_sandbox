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
        description="Automatically include isolated modules during orphan scans (enabled by default).",
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
    roi_threshold: float | None = Field(None, env="ROI_THRESHOLD")
    synergy_threshold: float | None = Field(None, env="SYNERGY_THRESHOLD")
    roi_confidence: float | None = Field(None, env="ROI_CONFIDENCE")
    synergy_confidence: float | None = Field(None, env="SYNERGY_CONFIDENCE")
    min_integration_roi: float = Field(
        0.0,
        env="MIN_INTEGRATION_ROI",
        description="Minimum ROI increase required for module auto-integration.",
    )
    synergy_threshold_window: int | None = Field(None, env="SYNERGY_THRESHOLD_WINDOW")
    synergy_threshold_weight: float | None = Field(None, env="SYNERGY_THRESHOLD_WEIGHT")
    synergy_ma_window: int | None = Field(None, env="SYNERGY_MA_WINDOW")
    synergy_stationarity_confidence: float | None = Field(None, env="SYNERGY_STATIONARITY_CONFIDENCE")
    synergy_std_threshold: float | None = Field(None, env="SYNERGY_STD_THRESHOLD")
    synergy_variance_confidence: float | None = Field(None, env="SYNERGY_VARIANCE_CONFIDENCE")

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
    scenario_metric_thresholds: dict[str, float] = Field(
        default_factory=dict,
        env="SCENARIO_METRIC_THRESHOLDS",
        description="Thresholds for scenario-specific metrics returned by _scenario_specific_metrics.",
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
            "Risk scores at or above this value raise non-blocking warnings. The "
            "default of 0.5 balances sensitivity with noise."
        ),
    )
    alignment_failure_threshold: float = Field(
        0.9,
        env="ALIGNMENT_FAILURE_THRESHOLD",
        description=(
            "Risk scores at or above this value are considered severe. A high "
            "default of 0.9 avoids false positives while still flagging critical "
            "issues."
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
