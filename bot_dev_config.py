from __future__ import annotations

"""Configuration for :class:`BotDevelopmentBot`."""

from dataclasses import dataclass, field
from pathlib import Path
import os
import logging
from typing import List, Callable, Dict, Tuple


@dataclass
class BotDevConfig:
    """Central configuration for bot development."""

    repo_base: Path = field(
        default_factory=lambda: Path(os.getenv("BOT_DEV_REPO_BASE", "dev_repos"))
    )
    es_url: str | None = os.getenv("BOT_DEV_ES_URL")
    es_index: str = os.getenv("BOT_DEV_ES_INDEX", "patterns")
    desktop_url: str = os.getenv("VISUAL_DESKTOP_URL", "http://127.0.0.1:8001")
    laptop_url: str = os.getenv("VISUAL_LAPTOP_URL", "http://127.0.0.1:8002")
    visual_agent_urls: List[str] = field(
        default_factory=lambda: (
            os.getenv("VISUAL_AGENT_URLS")
            or ""  # sentinel so split works
        ).split(";")
    )
    visual_agent_token: str = field(default_factory=lambda: os.getenv("VISUAL_AGENT_TOKEN", ""))
    visual_token_refresh_cmd: str | None = os.getenv("VISUAL_TOKEN_REFRESH_CMD")
    denial_phrases: List[str] = field(
        default_factory=lambda: os.getenv(
            "BOT_DEV_DENIAL_PHRASES",
            "i'm sorry;i am sorry;can't help with that",
        ).split(";")
    )
    headless: bool = os.getenv("BOT_DEV_HEADLESS", "0") == "1"
    default_templates: Dict[str, List[str]] = field(default_factory=dict)
    ocr_region: Tuple[int, int, int, int] = field(
        default_factory=lambda: (
            int(os.getenv("OCR_LEFT", "0")),
            int(os.getenv("OCR_TOP", "0")),
            int(os.getenv("OCR_WIDTH", "1920")),
            int(os.getenv("OCR_HEIGHT", "1080")),
        )
    )
    error_sinks: List[Callable[[str, str], None]] = field(default_factory=list)
    concurrency_workers: int = int(os.getenv("BOT_DEV_CONCURRENCY", "1"))
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
    visual_agent_poll_interval: float = float(os.getenv("VISUAL_AGENT_POLL_INTERVAL", "5"))

    def validate(self) -> None:
        """Warn if important settings are missing."""
        if not self.visual_agent_token:
            logging.getLogger("BotDev").warning("VISUAL_AGENT_TOKEN not set")
        if not any(self.visual_agent_urls):
            self.visual_agent_urls = [self.desktop_url, self.laptop_url]
        if self.concurrency_workers < 1:
            self.concurrency_workers = 1
        if not self.es_index:
            self.es_index = "patterns"
