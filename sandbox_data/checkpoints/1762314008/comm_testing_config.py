from __future__ import annotations

from pathlib import Path
import os
from pydantic_settings_compat import (
    BaseSettings,
    PYDANTIC_V2,
    SettingsConfigDict,
)
from pydantic import Field


class CommTestingSettings(BaseSettings):
    """Configuration for :mod:`communication_testing_bot`."""

    benchmark_threshold: float = Field(0.5, env="COMMS_BENCHMARK_THRESHOLD")
    db_path: str = Field("comm_tests.db", env="COMM_TEST_DB_PATH")
    data_dir: str = Field("", env="MENACE_DATA_DIR")

    model_config = SettingsConfigDict(
        env_file=os.getenv("MENACE_ENV_FILE", ".env"),
        extra="ignore",
    )

    if not PYDANTIC_V2:
        class Config:  # pragma: no cover - fallback for pydantic<2
            env_file = os.getenv("MENACE_ENV_FILE", ".env")
            extra = "ignore"

    @property
    def resolved_db_path(self) -> Path:
        base = Path(self.data_dir) if self.data_dir else Path()
        db = Path(self.db_path)
        return db if db.is_absolute() else base / db


SETTINGS = CommTestingSettings()

__all__ = ["CommTestingSettings", "SETTINGS"]
