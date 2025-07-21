from __future__ import annotations

"""Shared logging utilities for Menace."""

import json
import logging
import logging.config
import os
from pathlib import Path
from typing import Any, Dict


_DEFAULT_LOG_CONFIG: Dict[str, Any] = {
    "version": 1,
    "formatters": {
        "default": {"format": "%(asctime)s %(levelname)s %(name)s: %(message)s"}
    },
    "handlers": {"console": {"class": "logging.StreamHandler", "formatter": "default"}},
    "root": {"level": "INFO", "handlers": ["console"]},
}


def setup_logging(config_path: str | None = None) -> None:
    """Configure logging from *config_path* or defaults."""
    path = Path(config_path or os.getenv("MENACE_LOGGING_CONFIG", ""))
    if path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as fh:
                cfg = json.load(fh)
            logging.config.dictConfig(cfg)
            return
        except Exception:
            pass
    logging.config.dictConfig(_DEFAULT_LOG_CONFIG)


_def_configured = False


def get_logger(name: str) -> logging.Logger:
    """Return logger configured via :func:`setup_logging`."""
    global _def_configured
    if not _def_configured and not logging.getLogger().handlers:
        setup_logging()
        _def_configured = True
    return logging.getLogger(name)


def log_record(**fields: Any) -> Dict[str, Any]:
    """Return *fields* without ``None`` values for structured logging."""
    return {k: v for k, v in fields.items() if v is not None}


__all__ = ["setup_logging", "get_logger", "log_record"]
