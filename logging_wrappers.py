"""Structured logging wrappers for callables."""

from __future__ import annotations

import functools
import inspect
import logging
import time
import traceback
from datetime import datetime
from typing import Any

from logging_utils import get_logger, log_record


_DEFAULT_CONFIG: dict[str, Any] = {
    "logger_name": "menace.logging_wrappers",
    "event_name": "function_call",
    "include_args": True,
    "include_return": True,
    "include_exception": True,
    "max_depth": 4,
    "max_items": 50,
    "max_string": 2000,
    "truncate_marker": "<truncated>",
}


def _truncate_string(value: str, max_string: int, truncate_marker: str) -> str:
    if max_string < 0:
        return value
    if len(value) <= max_string:
        return value
    if max_string <= len(truncate_marker):
        return truncate_marker[:max_string]
    return value[: max_string - len(truncate_marker)] + truncate_marker


def _sanitize(
    value: Any,
    *,
    max_depth: int,
    max_items: int,
    max_string: int,
    truncate_marker: str,
    _depth: int = 0,
    _seen: set[int] | None = None,
) -> Any:
    if _seen is None:
        _seen = set()

    if max_depth >= 0 and _depth >= max_depth:
        return truncate_marker

    if value is None or isinstance(value, (bool, int, float)):
        return value

    if isinstance(value, str):
        return _truncate_string(value, max_string, truncate_marker)

    if isinstance(value, bytes):
        return _truncate_string(repr(value), max_string, truncate_marker)

    value_id = id(value)
    if value_id in _seen:
        return "<recursion>"
    _seen.add(value_id)

    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda item: str(item[0]))
        truncated = False
        if max_items >= 0 and len(items) > max_items:
            items = items[:max_items]
            truncated = True
        sanitized: dict[str, Any] = {}
        for key, item in items:
            sanitized[str(key)] = _sanitize(
                item,
                max_depth=max_depth,
                max_items=max_items,
                max_string=max_string,
                truncate_marker=truncate_marker,
                _depth=_depth + 1,
                _seen=_seen,
            )
        if truncated:
            sanitized[truncate_marker] = truncate_marker
        return sanitized

    if isinstance(value, (list, tuple)):
        items = list(value)
        truncated = False
        if max_items >= 0 and len(items) > max_items:
            items = items[:max_items]
            truncated = True
        sanitized = [
            _sanitize(
                item,
                max_depth=max_depth,
                max_items=max_items,
                max_string=max_string,
                truncate_marker=truncate_marker,
                _depth=_depth + 1,
                _seen=_seen,
            )
            for item in items
        ]
        if truncated:
            sanitized.append(truncate_marker)
        return sanitized

    if isinstance(value, set):
        items = sorted(value, key=lambda item: repr(item))
        return _sanitize(
            items,
            max_depth=max_depth,
            max_items=max_items,
            max_string=max_string,
            truncate_marker=truncate_marker,
            _depth=_depth + 1,
            _seen=_seen,
        )

    if hasattr(value, "__dict__"):
        return _sanitize(
            value.__dict__,
            max_depth=max_depth,
            max_items=max_items,
            max_string=max_string,
            truncate_marker=truncate_marker,
            _depth=_depth + 1,
            _seen=_seen,
        )

    if hasattr(value, "__slots__"):
        slots = getattr(value, "__slots__")
        if isinstance(slots, str):
            slots = [slots]
        data: dict[str, Any] = {}
        for slot in slots:
            if hasattr(value, slot):
                data[str(slot)] = _sanitize(
                    getattr(value, slot),
                    max_depth=max_depth,
                    max_items=max_items,
                    max_string=max_string,
                    truncate_marker=truncate_marker,
                    _depth=_depth + 1,
                    _seen=_seen,
                )
        return data

    return {
        "type": type(value).__name__,
        "repr": _truncate_string(repr(value), max_string, truncate_marker),
    }


def wrap_with_logging(callable_obj, config: dict | None = None):
    """Return callable wrapped with structured logging."""

    if getattr(callable_obj, "__menace_logging_wrapped__", False):
        return callable_obj

    cfg = dict(_DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    logger = get_logger(cfg["logger_name"])
    event_name = cfg["event_name"]
    include_args = bool(cfg["include_args"])
    include_return = bool(cfg["include_return"])
    include_exception = bool(cfg["include_exception"])
    max_depth = int(cfg["max_depth"])
    max_items = int(cfg["max_items"])
    max_string = int(cfg["max_string"])
    truncate_marker = str(cfg["truncate_marker"])

    def wrapper(*args, **kwargs):
        start = time.monotonic()
        try:
            result = callable_obj(*args, **kwargs)
        except Exception as exc:
            duration = time.monotonic() - start
            record: dict[str, Any] = {
                "event": event_name,
                "function": getattr(callable_obj, "__name__", str(callable_obj)),
                "module": getattr(callable_obj, "__module__", None),
                "duration_s": duration,
                "timestamp": datetime.utcnow().isoformat(),
            }
            if include_args:
                record["args"] = _sanitize(
                    args,
                    max_depth=max_depth,
                    max_items=max_items,
                    max_string=max_string,
                    truncate_marker=truncate_marker,
                )
                record["kwargs"] = _sanitize(
                    kwargs,
                    max_depth=max_depth,
                    max_items=max_items,
                    max_string=max_string,
                    truncate_marker=truncate_marker,
                )
            if include_exception:
                record["exception"] = {
                    "type": exc.__class__.__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exception(
                        exc.__class__, exc, exc.__traceback__
                    ),
                }
            logger.log(logging.INFO, event_name, extra=log_record(**record))
            raise

        duration = time.monotonic() - start
        record = {
            "event": event_name,
            "function": getattr(callable_obj, "__name__", str(callable_obj)),
            "module": getattr(callable_obj, "__module__", None),
            "duration_s": duration,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if include_args:
            record["args"] = _sanitize(
                args,
                max_depth=max_depth,
                max_items=max_items,
                max_string=max_string,
                truncate_marker=truncate_marker,
            )
            record["kwargs"] = _sanitize(
                kwargs,
                max_depth=max_depth,
                max_items=max_items,
                max_string=max_string,
                truncate_marker=truncate_marker,
            )
        if include_return:
            sanitized_return = None
            if result is not None:
                sanitized_return = _sanitize(
                    result,
                    max_depth=max_depth,
                    max_items=max_items,
                    max_string=max_string,
                    truncate_marker=truncate_marker,
                )
            record["return_value"] = {
                "value": sanitized_return,
                "is_none": result is None,
            }
        logger.log(logging.INFO, event_name, extra=log_record(**record))
        return result

    functools.update_wrapper(wrapper, callable_obj)
    wrapper.__signature__ = inspect.signature(callable_obj)
    wrapper.__menace_logging_wrapped__ = True
    return wrapper


__all__ = ["wrap_with_logging"]
