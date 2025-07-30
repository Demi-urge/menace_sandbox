from __future__ import annotations

"""Shared logging utilities for Menace."""

import json
import logging
import logging.config
import os
import base64
from pathlib import Path
from typing import Any, Dict
from datetime import datetime
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


class _ModuleFilter(logging.Filter):
    """Filter records by logger name."""

    def __init__(self, modules: list[str]) -> None:
        super().__init__()
        self._mods = tuple(modules)

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - simple
        return record.name in self._mods or record.name.split(".")[0] in self._mods


class AuditTrailHandler(logging.Handler):
    """Log records to an :class:`AuditTrail`."""

    def __init__(self, trail: "AuditTrail") -> None:  # pragma: no cover - tiny
        super().__init__()
        self.trail = trail
        self.setFormatter(JSONFormatter())

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - tiny
        try:
            msg = self.format(record)
            self.trail.record(msg)
        except Exception:
            pass


class KafkaLogHandler(logging.Handler):
    """Publish log records via :class:`KafkaMetaLogger`."""

    def __init__(self, logger: "KafkaMetaLogger") -> None:  # pragma: no cover - tiny
        super().__init__()
        self.logger = logger

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - tiny
        try:
            payload = {
                "time": datetime.utcnow().isoformat(),
                "level": record.levelname,
                "name": record.name,
                "message": record.getMessage(),
            }
            self.logger.log(LogEvent("record", payload))
        except Exception:
            pass


def _central_handler() -> logging.Handler | None:
    """Return handler for SANDBOX_CENTRAL_LOGGING if configured."""
    if os.getenv("SANDBOX_CENTRAL_LOGGING") != "1":
        return None
    if os.getenv("KAFKA_HOSTS"):
        try:
            from meta_logging import KafkaMetaLogger, LogEvent
        except Exception:  # pragma: no cover - optional dependency
            return None
        km = KafkaMetaLogger(brokers=os.getenv("KAFKA_HOSTS"), topic_prefix="menace.logs")
        return KafkaLogHandler(km)
    try:
        from audit_trail import AuditTrail
    except Exception:  # pragma: no cover - optional dependency
        return None
    key_b64 = os.getenv("AUDIT_PRIVKEY")
    priv = base64.b64decode(key_b64) if key_b64 else None
    path = os.getenv("AUDIT_LOG_PATH", "audit.log")
    trail = AuditTrail(path, priv)
    return AuditTrailHandler(trail)


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


def setup_logging(config_path: str | None = None, level: str | int | None = None) -> None:
    """Configure logging from *config_path* or defaults.

    When *level* is provided the root logger level is overridden after
    configuration. The argument may be a logging level name or numeric value.
    If ``SANDBOX_DEBUG=1`` (or ``SANDBOX_VERBOSE=1`` for backward compatibility)
    is set and *level* is ``None`` the root logger level defaults to
    :data:`logging.DEBUG`."""
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
    if level is None and (
        os.getenv("SANDBOX_VERBOSE") == "1" or os.getenv("SANDBOX_DEBUG") == "1"
    ):
        logging.getLogger().setLevel(logging.DEBUG)
    elif level is not None:
        if isinstance(level, int):
            logging.getLogger().setLevel(level)
        else:
            logging.getLogger().setLevel(getattr(logging, str(level).upper(), logging.INFO))
    handler = _central_handler()
    if handler:
        handler.addFilter(
            _ModuleFilter(["SelfTestService", "SynergyAutoTrainer", "synergy_monitor"])
        )
        logging.getLogger().addHandler(handler)


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
    "AuditTrailHandler",
    "KafkaLogHandler",
]
