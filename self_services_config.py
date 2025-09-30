from __future__ import annotations

"""Pydantic models for self-learning and self-test service configuration."""

from pathlib import Path
from typing import Any

try:  # pragma: no cover - allow running as script
    from .dynamic_path_router import resolve_path  # type: ignore
except Exception:  # pragma: no cover - fallback when executed directly
    from dynamic_path_router import resolve_path  # type: ignore

try:  # pragma: no cover - compatibility with pydantic v1/v2
    from pydantic_settings import BaseSettings, SettingsConfigDict
    PYDANTIC_V2 = True
except Exception:  # pragma: no cover - fallback for pydantic<2
    from pydantic import BaseSettings  # type: ignore
    PYDANTIC_V2 = False
    SettingsConfigDict = dict  # type: ignore[misc]
from pydantic import Field
try:  # pragma: no cover - compatibility shim
    from pydantic import field_validator
except Exception:  # pragma: no cover
    from pydantic import validator as field_validator  # type: ignore

FIELD_VALIDATOR_KWARGS: dict[str, Any] = {}
if not PYDANTIC_V2:
    FIELD_VALIDATOR_KWARGS["allow_reuse"] = True


class SelfLearningConfig(BaseSettings):
    """Configuration for the self-learning service.

    Values may be provided via environment variables.  Direct paths must point
    to existing directories so the service can write its state without failing
    later in the run.
    """

    persist_events: Path | None = Field(
        default=None,
        validation_alias="SELF_LEARNING_PERSIST_EVENTS",
        description="Optional path where the event bus should persist state.",
    )
    persist_progress: Path | None = Field(
        default=None,
        validation_alias="SELF_LEARNING_PERSIST_PROGRESS",
        description="Optional path for storing evaluation results on shutdown.",
    )
    prune_interval: int = Field(
        default=50,
        validation_alias="PRUNE_INTERVAL",
        description="Number of new interactions before pruning GPT memory.",
    )

    @field_validator("prune_interval", **FIELD_VALIDATOR_KWARGS)
    @classmethod
    def _validate_prune_interval(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("prune_interval must be positive")
        return v

    @field_validator("persist_events", "persist_progress", **FIELD_VALIDATOR_KWARGS)
    @classmethod
    def _validate_parent_exists(cls, v: Path | None, info: Any) -> Path | None:
        if v is not None and not v.parent.exists():
            raise ValueError(
                f"{info.field_name} directory does not exist: {v.parent}"
            )
        return v

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")
    if not PYDANTIC_V2:  # pragma: no cover - pydantic v1 fallback
        class Config:  # type: ignore[no-redef]
            env_prefix = ""
            extra = "ignore"


class SelfTestConfig(BaseSettings):
    """Configuration for the self-test service."""

    lock_file: Path = Field(
        default=resolve_path("sandbox_data") / "self_test.lock",
        validation_alias="SELF_TEST_LOCK_FILE",
        description="File used to serialise self-test runs.",
    )
    report_dir: Path = Field(
        default=resolve_path("sandbox_data") / "self_test_reports",
        validation_alias="SELF_TEST_REPORT_DIR",
        description="Directory used to store self-test reports.",
    )

    @field_validator("lock_file", **FIELD_VALIDATOR_KWARGS)
    @classmethod
    def _validate_lock_parent(cls, v: Path) -> Path:
        if not v.parent.exists():
            raise ValueError(f"lock file directory does not exist: {v.parent}")
        return v

    @field_validator("report_dir", **FIELD_VALIDATOR_KWARGS)
    @classmethod
    def _validate_report_dir(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"report_dir does not exist: {v}")
        return v
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")
    if not PYDANTIC_V2:  # pragma: no cover - pydantic v1 fallback
        class Config:  # type: ignore[no-redef]
            env_prefix = ""
            extra = "ignore"


class RepoScanConfig(BaseSettings):
    """Configuration for repository scanning."""

    enabled: bool = Field(
        default=True,
        validation_alias="REPO_SCAN_ENABLED",
        description="Enable periodic repository scanning for enhancement suggestions.",
    )
    interval: int = Field(
        default=3600,
        validation_alias="REPO_SCAN_INTERVAL",
        description="Seconds between repository scans.",
    )

    @field_validator("interval", **FIELD_VALIDATOR_KWARGS)
    @classmethod
    def _validate_interval(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("interval must be positive")
        return v

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")
    if not PYDANTIC_V2:  # pragma: no cover - pydantic v1 fallback
        class Config:  # type: ignore[no-redef]
            env_prefix = ""
            extra = "ignore"


__all__ = ["SelfLearningConfig", "SelfTestConfig", "RepoScanConfig"]
