"""Structured logging wrapper for function calls."""

from __future__ import annotations

import logging
import traceback
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from functools import update_wrapper
from time import monotonic
from typing import Any

from menace.infra.logging import get_logger, log_event

_DEFAULT_CONFIG: dict[str, Any] = {
    "logger_name": "menace.infra.logging_wrapper",
    "event_prefix": "function",
    "include_return": True,
    "include_args": True,
    "include_kwargs": True,
    "max_collection_items": 25,
    "max_string_length": 200,
    "include_timestamp": False,
}


def wrap_with_logging(func, config: dict[str, Any] | None = None):
    """Return a logging wrapper for the given function.

    Args:
        func: Callable to wrap.
        config: Configuration overrides for logging behavior.

    Returns:
        Callable: Wrapped function with structured logging.
    """

    if getattr(func, "__menace_logging_wrapped__", False):
        return func

    resolved_config = dict(_DEFAULT_CONFIG)
    if config:
        resolved_config.update(config)

    logger = _resolve_logger(resolved_config)
    function_name = getattr(func, "__qualname__", getattr(func, "__name__", "<unknown>"))
    module_name = getattr(func, "__module__", "<unknown>")
    event_prefix = str(resolved_config.get("event_prefix", "function"))

    def wrapper(*args, **kwargs):
        base_context = {
            "function": function_name,
            "module": module_name,
        }
        if resolved_config.get("include_args", True):
            base_context["args"] = _serialize_value(
                list(args),
                resolved_config,
            )
        if resolved_config.get("include_kwargs", True):
            base_context["kwargs"] = _serialize_value(
                dict(kwargs),
                resolved_config,
            )
        _add_timestamp(base_context, resolved_config)
        log_event(logger, f"{event_prefix}.call", base_context)

        start_time = monotonic()
        try:
            result = func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            duration_ms = _duration_ms(start_time)
            error_context = dict(base_context)
            error_context["duration_ms"] = duration_ms
            error_context["exception_type"] = type(exc).__name__
            error_context["exception_message"] = _serialize_value(
                exc,
                resolved_config,
            )
            error_context["traceback"] = _serialize_value(
                _format_traceback(exc),
                resolved_config,
            )
            log_event(logger, f"{event_prefix}.exception", error_context)
            raise

        duration_ms = _duration_ms(start_time)
        return_context = dict(base_context)
        return_context["duration_ms"] = duration_ms
        if resolved_config.get("include_return", True):
            return_context["result"] = _serialize_value(result, resolved_config)
        log_event(logger, f"{event_prefix}.return", return_context)
        return result

    wrapper.__wrapped__ = func
    wrapper.__menace_logging_wrapped__ = True
    update_wrapper(wrapper, func)
    return wrapper


def _resolve_logger(config: Mapping[str, Any]) -> logging.Logger:
    name = config.get("logger_name", "menace.infra.logging_wrapper")
    return get_logger(str(name))["data"]["logger"]


def _add_timestamp(context: dict[str, Any], config: Mapping[str, Any]) -> None:
    if config.get("include_timestamp", False):
        context["timestamp"] = datetime.now(timezone.utc).isoformat()


def _duration_ms(start_time: float) -> float:
    return round((monotonic() - start_time) * 1000.0, 3)


def _serialize_value(value: Any, config: Mapping[str, Any], _seen: set[int] | None = None) -> Any:
    if _seen is None:
        _seen = set()
    value_id = id(value)
    if value_id in _seen:
        return "<recursion>"
    _seen.add(value_id)

    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _truncate_string(value, config)
    if isinstance(value, (bytes, bytearray)):
        return _truncate_string(repr(value), config)
    if isinstance(value, Mapping):
        return _serialize_mapping(value, config, _seen)
    if isinstance(value, Sequence):
        return _serialize_sequence(list(value), config, _seen)
    if isinstance(value, (set, frozenset)):
        return _serialize_unordered(value, config, _seen)

    return _truncate_string(_safe_repr(value), config)


def _serialize_mapping(
    value: Mapping[Any, Any],
    config: Mapping[str, Any],
    _seen: set[int],
) -> dict[str, Any]:
    items = [
        (_truncate_string(_safe_key(key), config), item_value)
        for key, item_value in value.items()
    ]
    items = sorted(items, key=lambda pair: pair[0])
    max_items = int(config.get("max_collection_items", 25))
    serialized: dict[str, Any] = {}
    for key, item_value in items[:max_items]:
        serialized[key] = _serialize_value(item_value, config, _seen)
    return serialized


def _serialize_sequence(
    value: Sequence[Any],
    config: Mapping[str, Any],
    _seen: set[int],
) -> list[Any]:
    max_items = int(config.get("max_collection_items", 25))
    limited = list(value)[:max_items]
    return [_serialize_value(item, config, _seen) for item in limited]


def _serialize_unordered(
    value: set[Any] | frozenset[Any],
    config: Mapping[str, Any],
    _seen: set[int],
) -> list[Any]:
    serialized_items = [_serialize_value(item, config, _seen) for item in value]
    serialized_items = sorted(serialized_items, key=lambda item: _safe_sort_key(item))
    max_items = int(config.get("max_collection_items", 25))
    return serialized_items[:max_items]


def _safe_key(value: Any) -> str:
    return _safe_repr(value)


def _safe_repr(value: Any) -> str:
    try:
        return repr(value)
    except Exception:  # noqa: BLE001
        try:
            return str(value)
        except Exception:  # noqa: BLE001
            return "<unrepresentable>"


def _safe_sort_key(value: Any) -> str:
    try:
        return _safe_repr(value)
    except Exception:  # noqa: BLE001
        return "<unrepresentable>"


def _truncate_string(value: str, config: Mapping[str, Any]) -> str:
    max_length = int(config.get("max_string_length", 200))
    if max_length < 0:
        return value
    if len(value) <= max_length:
        return value
    if max_length <= 3:
        return value[:max_length]
    return f"{value[:max_length - 3]}..."


def _format_traceback(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
