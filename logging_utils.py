from __future__ import annotations

"""Shared logging utilities for Menace."""

import json
import logging
import logging.config
import os
import base64
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Dict
from datetime import datetime
import contextvars
import importlib


_correlation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "correlation_id", default=None
)


def _config_mod():
    """Return the configuration module regardless of import style."""
    try:  # pragma: no cover - package installed
        return importlib.import_module("menace.config")
    except Exception:  # pragma: no cover - fallback to local import
        return importlib.import_module("config")


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


class _LockedHandlerMixin:
    """Mixin providing cross-process file locking for handlers."""

    def __init__(self, filename: str, *args, **kwargs) -> None:
        from lock_utils import SandboxLock

        self._file_lock = SandboxLock(f"{filename}.lock")
        super().__init__(filename, *args, **kwargs)

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - simple
        with self._file_lock:
            if hasattr(self, "shouldRollover") and self.shouldRollover(record):  # type: ignore[attr-defined]
                self.doRollover()  # type: ignore[attr-defined]
            logging.FileHandler.emit(self, record)


class LockedRotatingFileHandler(_LockedHandlerMixin, RotatingFileHandler):
    """Size-based rotating file handler using :class:`SandboxLock`."""

    pass


class LockedTimedRotatingFileHandler(_LockedHandlerMixin, TimedRotatingFileHandler):
    """Time-based rotating file handler using :class:`SandboxLock`."""

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
    If ``SANDBOX_VERBOSE=1`` (or ``SANDBOX_DEBUG=1`` for compatibility) is set
    and *level* is ``None`` the root logger level defaults to
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
    cfg_mod = _config_mod()
    if level is None:
        level_name = cfg_mod.get_config().logging.verbosity
        level = getattr(logging, level_name.upper(), logging.INFO)
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

    bus = getattr(cfg_mod, "_EVENT_BUS", None)
    if bus:
        bus.subscribe("config.reload", lambda *_: _apply_log_level())


def _apply_log_level() -> None:
    """Apply current config logging verbosity to root logger."""
    cfg_mod = _config_mod()
    level_name = cfg_mod.get_config().logging.verbosity
    logging.getLogger().setLevel(getattr(logging, level_name.upper(), logging.INFO))


_def_configured = False


def get_logger(name: str) -> logging.Logger:
    """Return logger configured via :func:`setup_logging`."""
    global _def_configured
    if not _def_configured and not logging.getLogger().handlers:
        setup_logging()
        _def_configured = True
    return logging.getLogger(name)


_RESERVED_LOG_ATTRS = set(
    logging.LogRecord(
        name="", level=logging.INFO, pathname="", lineno=0, msg="", args=(), exc_info=None
    ).__dict__
)
# These are injected by the logging framework during formatting.
_RESERVED_LOG_ATTRS.update({"message", "asctime"})


def _safe_key(key: str, existing: Dict[str, Any]) -> str:
    """Return a key that will not clash with :class:`logging.LogRecord` fields."""

    if key not in _RESERVED_LOG_ATTRS and key not in existing:
        return key

    base = f"extra_{key}"
    if base not in _RESERVED_LOG_ATTRS and base not in existing:
        return base

    # Fall back to a numeric suffix if our preferred alias is still unsafe.
    idx = 1
    candidate = f"{base}_{idx}"
    while candidate in _RESERVED_LOG_ATTRS or candidate in existing:
        idx += 1
        candidate = f"{base}_{idx}"
    return candidate


def log_record(**fields: Any) -> Dict[str, Any]:
    """Return *fields* sanitized for use with ``Logger.extra``."""

    safe: Dict[str, Any] = {}
    for key, value in fields.items():
        if value is None:
            continue
        safe[_safe_key(key, safe)] = value
    return safe


__all__ = [
    "setup_logging",
    "get_logger",
    "log_record",
    "JSONFormatter",
    "set_correlation_id",
    "CorrelationIDFilter",
    "AuditTrailHandler",
    "KafkaLogHandler",
    "LockedRotatingFileHandler",
    "LockedTimedRotatingFileHandler",
]
