"""Structured logging wrapper for function calls.

Invariants:
- Wrapping does not mutate input arguments, return values, or configuration.
- Wrapped call behavior is preserved (results/exceptions are unchanged).
- Logged payloads are deterministic and JSON-safe.
"""

from __future__ import annotations

import inspect
import json
import logging
import traceback
from collections.abc import Mapping, Sequence, Set
from datetime import datetime, timezone
from functools import update_wrapper
from time import monotonic
from typing import Any

from menace.infra.logging import get_logger, log_event

_DEFAULT_CONFIG: dict[str, Any] = {
    "logger_name": "menace.infra.logging_wrapper",
    "event_start": "function.call",
    "event_success": "function.return",
    "event_failure": "function.exception",
    "max_depth": 4,
    "max_items": 25,
    "max_str_len": 200,
    "log_args": True,
    "log_kwargs": True,
    "log_return": True,
    "include_timestamp": False,
}


def wrap_with_logging(func, config: dict[str, Any] | None = None):
    """Return a logging wrapper for the given function.

    Args:
        func: Callable to wrap.
        config: Configuration overrides for logging behavior.

    Returns:
        Callable: Wrapped function with structured logging.

    Raises:
        ValueError: If configuration is invalid or the function is already wrapped.
    """

    if getattr(func, "__menace_logging_wrapped__", False):
        raise ValueError("Function is already wrapped for logging.")
    if not callable(func):
        raise ValueError("wrap_with_logging expects a callable.")

    resolved_config = _merge_config(config)
    logger = _resolve_logger(resolved_config)
    signature = inspect.signature(func)
    function_name = getattr(func, "__qualname__", getattr(func, "__name__", "<unknown>"))
    module_name = getattr(func, "__module__", "<unknown>")

    def wrapper(*args, **kwargs):
        base_context = {
            "function": function_name,
            "module": module_name,
        }
        truncation: dict[str, list[dict[str, Any]]] = {
            "strings": [],
            "collections": [],
            "depth": [],
        }
        if resolved_config["log_args"]:
            base_context["args"] = _normalize_value(
                list(args),
                max_depth=resolved_config["max_depth"],
                max_items=resolved_config["max_items"],
                max_str_len=resolved_config["max_str_len"],
                truncation=truncation,
                path="args",
            )
        if resolved_config["log_kwargs"]:
            base_context["kwargs"] = _normalize_value(
                dict(kwargs),
                max_depth=resolved_config["max_depth"],
                max_items=resolved_config["max_items"],
                max_str_len=resolved_config["max_str_len"],
                truncation=truncation,
                path="kwargs",
            )
        _add_timestamp(base_context, resolved_config)
        if _has_truncation(truncation):
            base_context["truncation"] = _compact_truncation(truncation)
        log_event(logger, resolved_config["event_start"], base_context)

        start_time = monotonic()
        try:
            result = func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            duration_s, duration_ms = _duration(start_time)
            error_context = dict(base_context)
            error_context["duration_s"] = duration_s
            error_context["duration_ms"] = duration_ms
            error_context["exception"] = _normalize_exception(
                exc,
                resolved_config,
                truncation,
            )
            error_context["traceback"] = _normalize_value(
                _format_traceback(exc),
                max_depth=resolved_config["max_depth"],
                max_items=resolved_config["max_items"],
                max_str_len=resolved_config["max_str_len"],
                truncation=truncation,
                path="traceback",
            )
            if _has_truncation(truncation):
                error_context["truncation"] = _compact_truncation(truncation)
            log_event(logger, resolved_config["event_failure"], error_context)
            raise

        duration_s, duration_ms = _duration(start_time)
        return_context = dict(base_context)
        return_context["duration_s"] = duration_s
        return_context["duration_ms"] = duration_ms
        if resolved_config["log_return"]:
            return_context["result"] = _normalize_value(
                result,
                max_depth=resolved_config["max_depth"],
                max_items=resolved_config["max_items"],
                max_str_len=resolved_config["max_str_len"],
                truncation=truncation,
                path="result",
            )
        if _has_truncation(truncation):
            return_context["truncation"] = _compact_truncation(truncation)
        log_event(logger, resolved_config["event_success"], return_context)
        return result

    update_wrapper(wrapper, func)
    wrapper.__wrapped__ = func
    wrapper.__signature__ = signature
    wrapper.__menace_logging_wrapped__ = True
    return wrapper


def _merge_config(config: dict[str, Any] | None) -> dict[str, Any]:
    resolved = dict(_DEFAULT_CONFIG)
    if config is None:
        return resolved
    if not isinstance(config, dict):
        raise ValueError("config must be a dict when provided.")
    resolved.update(config)
    _validate_config(resolved)
    return resolved


def _validate_config(config: Mapping[str, Any]) -> None:
    _ensure_type(config.get("logger_name"), str, "logger_name")
    _ensure_type(config.get("event_start"), str, "event_start")
    _ensure_type(config.get("event_success"), str, "event_success")
    _ensure_type(config.get("event_failure"), str, "event_failure")
    _ensure_bool(config.get("log_args"), "log_args")
    _ensure_bool(config.get("log_kwargs"), "log_kwargs")
    _ensure_bool(config.get("log_return"), "log_return")
    _ensure_bool(config.get("include_timestamp"), "include_timestamp")
    _ensure_int(config.get("max_depth"), "max_depth")
    _ensure_int(config.get("max_items"), "max_items")
    _ensure_int(config.get("max_str_len"), "max_str_len")


def _ensure_type(value: Any, expected: type, label: str) -> None:
    if not isinstance(value, expected):
        raise ValueError(f"{label} must be a {expected.__name__}.")


def _ensure_bool(value: Any, label: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a bool.")


def _ensure_int(value: Any, label: str) -> None:
    if not isinstance(value, int):
        raise ValueError(f"{label} must be an int.")


def _resolve_logger(config: Mapping[str, Any]) -> logging.Logger:
    name = config.get("logger_name", "menace.infra.logging_wrapper")
    return get_logger(str(name))["data"]["logger"]


def _add_timestamp(context: dict[str, Any], config: Mapping[str, Any]) -> None:
    if config.get("include_timestamp", False):
        context["timestamp"] = datetime.now(timezone.utc).isoformat()


def _duration(start_time: float) -> tuple[float, float]:
    duration_s = monotonic() - start_time
    duration_ms = round(duration_s * 1000.0, 3)
    return round(duration_s, 6), duration_ms


def _format_traceback(exc: BaseException) -> list[str]:
    return [line.rstrip("\n") for line in traceback.format_exception(type(exc), exc, exc.__traceback__)]


def _normalize_exception(
    exc: BaseException,
    config: Mapping[str, Any],
    truncation: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    normalized_message = _normalize_value(
        str(exc),
        max_depth=config["max_depth"],
        max_items=config["max_items"],
        max_str_len=config["max_str_len"],
        truncation=truncation,
        path="exception.message",
    )
    return {
        "type": type(exc).__name__,
        "message": normalized_message,
    }


def _normalize_value(
    value: Any,
    *,
    max_depth: int,
    max_items: int,
    max_str_len: int,
    truncation: dict[str, list[dict[str, Any]]],
    path: str,
    _depth: int = 0,
    _seen: set[int] | None = None,
) -> Any:
    if _seen is None:
        _seen = set()

    if max_depth >= 0 and _depth > max_depth:
        truncation["depth"].append({"path": path, "max_depth": max_depth})
        return _repr_with_type(value, max_str_len, truncation, f"{path}.depth")

    value_id = id(value)
    if value_id in _seen:
        return {"type": "recursion", "repr": "<recursion>"}
    _seen.add(value_id)

    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _truncate_string(value, max_str_len, truncation, path)
    if isinstance(value, (bytes, bytearray)):
        return _repr_with_type(value, max_str_len, truncation, path)
    if isinstance(value, Mapping):
        return _normalize_mapping(
            value,
            max_depth,
            max_items,
            max_str_len,
            truncation,
            path,
            _depth,
            _seen,
        )
    if isinstance(value, Set):
        return _normalize_set(
            value,
            max_depth,
            max_items,
            max_str_len,
            truncation,
            path,
            _depth,
            _seen,
        )
    if isinstance(value, Sequence):
        return _normalize_sequence(
            value,
            max_depth,
            max_items,
            max_str_len,
            truncation,
            path,
            _depth,
            _seen,
        )

    return _repr_with_type(value, max_str_len, truncation, path)


def _normalize_mapping(
    value: Mapping[Any, Any],
    max_depth: int,
    max_items: int,
    max_str_len: int,
    truncation: dict[str, list[dict[str, Any]]],
    path: str,
    _depth: int,
    _seen: set[int],
) -> dict[str, Any]:
    items = [(_stable_key(key, max_str_len, truncation, path), val) for key, val in value.items()]
    items = sorted(items, key=lambda pair: pair[0])
    if max_items >= 0 and len(items) > max_items:
        truncation["collections"].append(
            {"path": path, "max_items": max_items, "original_length": len(items)}
        )
    limited_items = items if max_items < 0 else items[:max_items]
    normalized: dict[str, Any] = {}
    for key, item_value in limited_items:
        normalized[key] = _normalize_value(
            item_value,
            max_depth=max_depth,
            max_items=max_items,
            max_str_len=max_str_len,
            truncation=truncation,
            path=f"{path}.{key}",
            _depth=_depth + 1,
            _seen=_seen,
        )
    return normalized


def _normalize_sequence(
    value: Sequence[Any],
    max_depth: int,
    max_items: int,
    max_str_len: int,
    truncation: dict[str, list[dict[str, Any]]],
    path: str,
    _depth: int,
    _seen: set[int],
) -> list[Any]:
    items = list(value)
    if max_items >= 0 and len(items) > max_items:
        truncation["collections"].append(
            {"path": path, "max_items": max_items, "original_length": len(items)}
        )
    limited_items = items if max_items < 0 else items[:max_items]
    return [
        _normalize_value(
            item,
            max_depth=max_depth,
            max_items=max_items,
            max_str_len=max_str_len,
            truncation=truncation,
            path=f"{path}[{index}]",
            _depth=_depth + 1,
            _seen=_seen,
        )
        for index, item in enumerate(limited_items)
    ]


def _normalize_set(
    value: Set[Any],
    max_depth: int,
    max_items: int,
    max_str_len: int,
    truncation: dict[str, list[dict[str, Any]]],
    path: str,
    _depth: int,
    _seen: set[int],
) -> list[Any]:
    items = list(value)
    if max_items >= 0 and len(items) > max_items:
        truncation["collections"].append(
            {"path": path, "max_items": max_items, "original_length": len(items)}
        )
    limited_items = items if max_items < 0 else items[:max_items]
    normalized_items = [
        _normalize_value(
            item,
            max_depth=max_depth,
            max_items=max_items,
            max_str_len=max_str_len,
            truncation=truncation,
            path=f"{path}[]",
            _depth=_depth + 1,
            _seen=_seen,
        )
        for item in limited_items
    ]
    normalized_items.sort(key=_stable_sort_key)
    return normalized_items


def _stable_key(
    value: Any,
    max_str_len: int,
    truncation: dict[str, list[dict[str, Any]]],
    path: str,
) -> str:
    rep = _safe_repr(value)
    rep = _truncate_string(rep, max_str_len, truncation, f"{path}.key")
    return f"{type(value).__name__}:{rep}"


def _stable_sort_key(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return _safe_repr(value)


def _repr_with_type(
    value: Any,
    max_str_len: int,
    truncation: dict[str, list[dict[str, Any]]],
    path: str,
) -> dict[str, str]:
    rep = _safe_repr(value)
    rep = _truncate_string(rep, max_str_len, truncation, f"{path}.repr")
    return {"type": type(value).__name__, "repr": rep}


def _safe_repr(value: Any) -> str:
    try:
        return repr(value)
    except Exception:  # noqa: BLE001
        try:
            return str(value)
        except Exception:  # noqa: BLE001
            return "<unrepresentable>"


def _truncate_string(
    value: str,
    max_length: int,
    truncation: dict[str, list[dict[str, Any]]],
    path: str,
) -> str:
    if max_length < 0 or len(value) <= max_length:
        return value
    truncation["strings"].append(
        {"path": path, "max_length": max_length, "original_length": len(value)}
    )
    if max_length <= 3:
        return value[:max_length]
    return f"{value[:max_length - 3]}..."


def _has_truncation(truncation: dict[str, list[dict[str, Any]]]) -> bool:
    return any(truncation.values())


def _compact_truncation(truncation: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {key: list(values) for key, values in truncation.items() if values}
