"""Logging wrapper utilities.

Configuration keys (defaults):
- logger_name: "menace.logging_wrapper"
- event_name: "function_call"
- include_args: True
- include_return: True
- include_exception: True
- max_depth: 4
- max_items: 50
- max_string: 2000
- truncate_marker: "<truncated>"
"""

from __future__ import annotations

from dataclasses import is_dataclass, fields
from datetime import datetime
import functools
import inspect
import logging
import time
import traceback
from collections.abc import Mapping, Sequence
from typing import Any

from logging_utils import get_logger, log_record


_DEFAULT_CONFIG: dict[str, Any] = {
    "logger_name": "menace.logging_wrapper",
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


def _sanitize_mapping(
    mapping: Mapping[Any, Any],
    *,
    max_depth: int,
    max_items: int,
    max_string: int,
    truncate_marker: str,
    _depth: int,
    _seen: set[int],
) -> dict[str, Any]:
    items = list(mapping.items())
    items.sort(key=lambda item: str(item[0]))
    if max_items >= 0 and len(items) > max_items:
        items = items[:max_items]
        truncated = True
    else:
        truncated = False
    sanitized: dict[str, Any] = {}
    for key, value in items:
        key_str = str(key)
        sanitized[key_str] = _sanitize(
            value,
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


def _sanitize_sequence(
    sequence: Sequence[Any],
    *,
    max_depth: int,
    max_items: int,
    max_string: int,
    truncate_marker: str,
    _depth: int,
    _seen: set[int],
) -> list[Any]:
    items = list(sequence)
    if max_items >= 0 and len(items) > max_items:
        items = items[:max_items]
        truncated = True
    else:
        truncated = False
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


def _sanitize_dataclass(
    obj: Any,
    *,
    max_depth: int,
    max_items: int,
    max_string: int,
    truncate_marker: str,
    _depth: int,
    _seen: set[int],
) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for field in fields(obj):
        data[field.name] = _sanitize(
            getattr(obj, field.name),
            max_depth=max_depth,
            max_items=max_items,
            max_string=max_string,
            truncate_marker=truncate_marker,
            _depth=_depth + 1,
            _seen=_seen,
        )
    return data


def _sanitize_object(
    obj: Any,
    *,
    max_depth: int,
    max_items: int,
    max_string: int,
    truncate_marker: str,
    _depth: int,
    _seen: set[int],
) -> dict[str, Any]:
    data: dict[str, Any] = {}
    if hasattr(obj, "__dict__"):
        data.update(
            _sanitize_mapping(
                obj.__dict__,
                max_depth=max_depth,
                max_items=max_items,
                max_string=max_string,
                truncate_marker=truncate_marker,
                _depth=_depth + 1,
                _seen=_seen,
            )
        )
    if hasattr(obj, "__slots__"):
        slots = getattr(obj, "__slots__")
        if isinstance(slots, str):
            slots = [slots]
        for slot in slots:
            if hasattr(obj, slot):
                data[str(slot)] = _sanitize(
                    getattr(obj, slot),
                    max_depth=max_depth,
                    max_items=max_items,
                    max_string=max_string,
                    truncate_marker=truncate_marker,
                    _depth=_depth + 1,
                    _seen=_seen,
                )
    return data


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

    if isinstance(value, Mapping):
        return _sanitize_mapping(
            value,
            max_depth=max_depth,
            max_items=max_items,
            max_string=max_string,
            truncate_marker=truncate_marker,
            _depth=_depth,
            _seen=_seen,
        )

    if isinstance(value, (list, tuple)):
        return _sanitize_sequence(
            value,
            max_depth=max_depth,
            max_items=max_items,
            max_string=max_string,
            truncate_marker=truncate_marker,
            _depth=_depth,
            _seen=_seen,
        )

    if isinstance(value, set):
        items = sorted(value, key=lambda item: repr(item))
        return _sanitize_sequence(
            items,
            max_depth=max_depth,
            max_items=max_items,
            max_string=max_string,
            truncate_marker=truncate_marker,
            _depth=_depth,
            _seen=_seen,
        )

    if is_dataclass(value):
        return _sanitize_dataclass(
            value,
            max_depth=max_depth,
            max_items=max_items,
            max_string=max_string,
            truncate_marker=truncate_marker,
            _depth=_depth,
            _seen=_seen,
        )

    if hasattr(value, "__dict__") or hasattr(value, "__slots__"):
        return _sanitize_object(
            value,
            max_depth=max_depth,
            max_items=max_items,
            max_string=max_string,
            truncate_marker=truncate_marker,
            _depth=_depth,
            _seen=_seen,
        )

    return _truncate_string(repr(value), max_string, truncate_marker)


def wrap_with_logging(callable_obj, config: dict | None = None):
    """Return callable wrapped with structured logging."""
    if getattr(callable_obj, "__menace_logging_wrapped__", False):
        return callable_obj

    cfg = dict(_DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    logger_name = cfg["logger_name"]
    event_name = cfg["event_name"]
    include_args = bool(cfg["include_args"])
    include_return = bool(cfg["include_return"])
    include_exception = bool(cfg["include_exception"])
    max_depth = int(cfg["max_depth"])
    max_items = int(cfg["max_items"])
    max_string = int(cfg["max_string"])
    truncate_marker = str(cfg["truncate_marker"])

    logger = get_logger(logger_name)

    def wrapper(*args, **kwargs):
        start = time.monotonic()
        try:
            result = callable_obj(*args, **kwargs)
        except Exception as exc:
            duration = time.monotonic() - start
            record: dict[str, Any] = {
                "event": event_name,
                "function": callable_obj.__name__,
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
            logger.log(
                logging.INFO,
                event_name,
                extra=log_record(**record),
            )
            raise

        duration = time.monotonic() - start
        record = {
            "event": event_name,
            "function": callable_obj.__name__,
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
            if result is None:
                record["return_value"] = {"value": None, "is_none": True}
            else:
                record["return_value"] = _sanitize(
                    result,
                    max_depth=max_depth,
                    max_items=max_items,
                    max_string=max_string,
                    truncate_marker=truncate_marker,
                )
        logger.log(
            logging.INFO,
            event_name,
            extra=log_record(**record),
        )
        return result

    functools.update_wrapper(wrapper, callable_obj)
    wrapper.__signature__ = inspect.signature(callable_obj)
    wrapper.__menace_logging_wrapped__ = True
    return wrapper


__all__ = ["wrap_with_logging"]
