from __future__ import annotations

"""Shared logging utilities for Menace."""

import json
import logging
import logging.config
import os
from pathlib import Path
from typing import Any, Dict
import contextvars


_correlation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "correlation_id", default=None
)


class CorrelationIDFilter(logging.Filter):
    """Inject correlation ID into log records if present."""

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - tiny
        cid = _correlation_id.get(None)
        record.correlation_id = cid
        return True


def set_correlation_id(cid: str | None) -> None:
    """Bind ``cid`` to the current context for log correlation."""
    _correlation_id.set(cid)


class JSONFormatter(logging.Formatter):
    """Format log records as compact JSON."""

    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - simple
        data = {
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        cid = getattr(record, "correlation_id", None)
        if cid:
            data["correlation_id"] = cid
        if record.exc_info:
            data["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(data)


_DEFAULT_LOG_CONFIG: Dict[str, Any] = {
    "version": 1,
    "formatters": {
        "default": {
            "format": "%(asctime)s %(levelname)s %(name)s [%(correlation_id)s]: %(message)s"
        },
        "json": {"()": "logging_utils.JSONFormatter"},
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "filters": ["correlation"],
        }
    },
    "root": {"level": "INFO", "handlers": ["console"]},
    "filters": {"correlation": {"()": "logging_utils.CorrelationIDFilter"}},
}


def setup_logging(config_path: str | None = None) -> None:
    """Configure logging from *config_path* or defaults."""
    path = Path(config_path or os.getenv("MENACE_LOGGING_CONFIG", ""))
    cfg: Dict[str, Any] | None = None
    if path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as fh:
                cfg = json.load(fh)
        except Exception:
            cfg = None
    if cfg is None:
        cfg = dict(_DEFAULT_LOG_CONFIG)
    if os.getenv("SANDBOX_JSON_LOGS") == "1":
        cfg = cfg.copy()
        cfg.setdefault("formatters", {})["json"] = {"()": "logging_utils.JSONFormatter"}
        for hname, handler in list(cfg.get("handlers", {}).items()):
            if isinstance(handler, dict) and hname == "console":
                handler = handler.copy()
                handler["formatter"] = "json"
                cfg["handlers"][hname] = handler
    logging.config.dictConfig(cfg)


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


__all__ = [
    "setup_logging",
    "get_logger",
    "log_record",
    "JSONFormatter",
    "set_correlation_id",
    "CorrelationIDFilter",
]
