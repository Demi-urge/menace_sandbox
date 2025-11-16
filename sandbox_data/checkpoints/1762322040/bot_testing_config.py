from __future__ import annotations

"""Configuration helpers for :mod:`bot_testing_bot`."""

import os
from pydantic_settings_compat import (
    BaseSettings,
    PYDANTIC_V2,
    SettingsConfigDict,
)
from pydantic import Field


class BotTestingSettings(BaseSettings):
    """Settings for :class:`~menace.bot_testing_bot.BotTestingBot`."""

    version: str = Field("0.1", env="BOT_TESTING_VERSION")
    db_backend: str = Field("sqlite", env="BOT_TESTING_DB_BACKEND")
    db_path: str = Field("testing_log.db", env="BOT_TESTING_DB_PATH")
    db_dsn: str = Field("dbname=testing_log", env="BOT_TESTING_DB_DSN")
    random_runs: int = Field(3, env="BOT_TESTING_RANDOM_RUNS")
    parallel: bool = Field(False, env="BOT_TESTING_PARALLEL")
    allow_unrandomized: bool = Field(True, env="BOT_TESTING_ALLOW_UNRANDOMIZED")
    db_write_attempts: int = Field(3, env="BOT_TESTING_DB_WRITE_ATTEMPTS")
    db_write_delay: float = Field(0.1, env="BOT_TESTING_DB_WRITE_DELAY")
    test_timeout: float | None = Field(None, env="BOT_TESTING_TIMEOUT")

    model_config = SettingsConfigDict(
        env_file=os.getenv("MENACE_ENV_FILE", ".env"),
        extra="ignore",
    )

    if not PYDANTIC_V2:
        class Config:  # pragma: no cover - fallback for pydantic<2
            env_file = os.getenv("MENACE_ENV_FILE", ".env")
            extra = "ignore"


__all__ = ["BotTestingSettings"]
