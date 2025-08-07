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
from pydantic import Field


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
    test_redundant_modules: bool = Field(
        True,
        env="SANDBOX_TEST_REDUNDANT",
        description="Integrate modules classified as redundant after validation.",
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
    sandbox_score_db: str = Field("score_history.db", env="SANDBOX_SCORE_DB")
    synergy_weights_lr: float = Field(0.1, env="SYNERGY_WEIGHTS_LR")

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


__all__ = ["SandboxSettings"]
