from __future__ import annotations

"""Configuration for :class:`BotDevelopmentBot`."""

from dataclasses import dataclass, field
from pathlib import Path
import os
from typing import List, Callable, Dict

from .error_flags import RAISE_ERRORS


@dataclass
class BotDevConfig:
    """Central configuration for bot development."""

    repo_base: Path = field(
        default_factory=lambda: Path(os.getenv("BOT_DEV_REPO_BASE", "dev_repos"))
    )
    es_url: str | None = os.getenv("BOT_DEV_ES_URL")
    es_index: str = os.getenv("BOT_DEV_ES_INDEX", "patterns")
    denial_phrases: List[str] = field(
        default_factory=lambda: os.getenv(
            "BOT_DEV_DENIAL_PHRASES",
            "i'm sorry;i am sorry;can't help with that",
        ).split(";")
    )
    headless: bool = os.getenv("BOT_DEV_HEADLESS", "0") == "1"
    default_templates: Dict[str, List[str]] = field(default_factory=dict)
    error_sinks: List[Callable[[str, str], None]] = field(default_factory=list)
    concurrency_workers: int = int(os.getenv("BOT_DEV_CONCURRENCY", "1"))
    raise_errors: bool = RAISE_ERRORS
    # retry/polling configuration
    file_write_attempts: int = int(os.getenv("FILE_WRITE_ATTEMPTS", "3"))
    file_write_retry_delay: float = float(os.getenv("FILE_WRITE_RETRY_DELAY", "0.5"))
    send_prompt_attempts: int = int(os.getenv("SEND_PROMPT_ATTEMPTS", "3"))
    send_prompt_retry_delay: float = float(os.getenv("SEND_PROMPT_RETRY_DELAY", "1.0"))
    engine_attempts: int = int(
        os.getenv("ENGINE_ATTEMPTS", os.getenv("GENERATION_ATTEMPTS", "3"))
    )
    engine_retry_delay: float = float(
        os.getenv("ENGINE_RETRY_DELAY", os.getenv("GENERATION_RETRY_DELAY", "1.0"))
    )
    engine_model: str = os.getenv(
        "ENGINE_MODEL", os.getenv("DEFAULT_MODEL", "internal-codex")
    )
    max_prompt_log_chars: int = int(os.getenv("MAX_PROMPT_LOG_CHARS", "200"))

    def validate(self) -> None:
        """Warn if important settings are missing."""
        self.concurrency_workers = max(1, self.concurrency_workers)
        if not self.es_index:
            self.es_index = "patterns"
        if "visual_agents" in self.__dict__:
            raise ValueError("visual_agents configuration is no longer supported")
