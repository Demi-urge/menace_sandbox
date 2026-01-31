"""Convenience wrappers for deterministic structured logging."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import functools
import logging
import time
import traceback
from typing import Any, TypeVar

T = TypeVar("T")

_WRAPPED_MARKER = "__menace_logging_wrapped__"

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
    items = sorted(mapping.items(), key=lambda item: str(item[0]))
    if max_items >= 0 and len(items) > max_items:
        items = items[:max_items]
        truncated = True
    else:
        truncated = False
    sanitized: dict[str, Any] = {}
    for key, value in items:
        sanitized[str(key)] = _sanitize(
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
    if hasattr(obj, "__dict__"):
        return _sanitize_mapping(
            obj.__dict__,
            max_depth=max_depth,
            max_items=max_items,
            max_string=max_string,
            truncate_marker=truncate_marker,
            _depth=_depth + 1,
            _seen=_seen,
        )
    if hasattr(obj, "__slots__"):
        slots = getattr(obj, "__slots__")
        if isinstance(slots, str):
            slots = [slots]
        data: dict[str, Any] = {}
        for slot in sorted(slots):
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
    return {"type": type(obj).__name__}


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

    return _sanitize_object(
        value,
        max_depth=max_depth,
        max_items=max_items,
        max_string=max_string,
        truncate_marker=truncate_marker,
        _depth=_depth,
        _seen=_seen,
    )


def wrap_with_logging(
    func: Callable[..., T],
    config: Mapping[str, Any] | None = None,
) -> Callable[..., T]:
    """Return callable wrapped with structured logging."""
    if getattr(func, _WRAPPED_MARKER, False):
        return func

    cfg = dict(_DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    logger_name = str(cfg["logger_name"])
    event_name = str(cfg["event_name"])
    include_args = bool(cfg["include_args"])
    include_return = bool(cfg["include_return"])
    include_exception = bool(cfg["include_exception"])
    max_depth = int(cfg["max_depth"])
    max_items = int(cfg["max_items"])
    max_string = int(cfg["max_string"])
    truncate_marker = str(cfg["truncate_marker"])

    logger = logging.getLogger(logger_name)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        start = time.monotonic()
        try:
            result = func(*args, **kwargs)
        except Exception as exc:
            duration = time.monotonic() - start
            record: dict[str, Any] = {
                "event": event_name,
                "function": func.__name__,
                "module": getattr(func, "__module__", None),
                "duration_s": duration,
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
            logger.info(event_name, extra={"payload": record})
            raise

        duration = time.monotonic() - start
        record = {
            "event": event_name,
            "function": func.__name__,
            "module": getattr(func, "__module__", None),
            "duration_s": duration,
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
        logger.info(event_name, extra={"payload": record})
        return result

    setattr(wrapper, _WRAPPED_MARKER, True)
    return wrapper


__all__ = ["wrap_with_logging"]
