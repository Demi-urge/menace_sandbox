"""Logging helpers for stabilization pipeline events."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import functools
import inspect
import logging
import time
import traceback as tb
import uuid
from typing import Any, Callable, Mapping, MutableMapping

from logging_utils import get_logger, log_record, set_correlation_id
from safe_serialization import sanitize_for_json


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


_DEFAULT_WRAP_CONFIG: dict[str, Any] = {
    "logger_name": "stabilization.logging_wrapper",
    "log_level": "INFO",
    "include_args": True,
    "include_return": True,
    "include_exceptions": True,
    "max_items": 25,
    "max_string_length": 200,
    "log_event_prefix": "stabilization.call.",
}


def _resolve_level(log_level: int | str) -> int:
    if isinstance(log_level, int):
        return log_level
    level = logging._nameToLevel.get(log_level.upper())
    if isinstance(level, int):
        return level
    return logging.INFO


def _truncate_string(value: str, max_length: int) -> tuple[str, int]:
    if max_length <= 0 or len(value) <= max_length:
        return value, 0
    return value[:max_length], len(value) - max_length


def _truncate_value(
    value: Any, *, max_items: int, max_string_length: int
) -> tuple[Any, int]:
    if isinstance(value, str):
        truncated, count = _truncate_string(value, max_string_length)
        return truncated, count
    if isinstance(value, Mapping):
        keys = sorted(value.keys())
        truncated_count = 0
        if max_items > 0 and len(keys) > max_items:
            truncated_count += len(keys) - max_items
            keys = keys[:max_items]
        truncated: dict[str, Any] = {}
        for key in keys:
            item, nested_count = _truncate_value(
                value[key], max_items=max_items, max_string_length=max_string_length
            )
            truncated[str(key)] = item
            truncated_count += nested_count
        return truncated, truncated_count
    if isinstance(value, list):
        truncated_count = 0
        items = value
        if max_items > 0 and len(items) > max_items:
            truncated_count += len(items) - max_items
            items = items[:max_items]
        truncated_items = []
        for item in items:
            clipped, nested_count = _truncate_value(
                item, max_items=max_items, max_string_length=max_string_length
            )
            truncated_items.append(clipped)
            truncated_count += nested_count
        return truncated_items, truncated_count
    if isinstance(value, tuple):
        truncated_list, truncated_count = _truncate_value(
            list(value), max_items=max_items, max_string_length=max_string_length
        )
        return tuple(truncated_list), truncated_count
    return value, 0


def _sanitize_and_truncate(
    value: Any, *, max_items: int, max_string_length: int
) -> tuple[Any, int]:
    sanitized = sanitize_for_json(value)
    truncated, truncated_count = _truncate_value(
        sanitized, max_items=max_items, max_string_length=max_string_length
    )
    return truncated, truncated_count


def _split_arguments(
    signature: inspect.Signature, bound: inspect.BoundArguments
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    args_list: list[dict[str, Any]] = []
    kwargs_dict: dict[str, Any] = {}
    for name, param in signature.parameters.items():
        if name not in bound.arguments:
            continue
        value = bound.arguments[name]
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
            args_list.append({"name": name, "value": value})
        elif param.kind is param.VAR_POSITIONAL:
            for item in value:
                args_list.append({"name": name, "value": item})
        elif param.kind is param.KEYWORD_ONLY:
            kwargs_dict[name] = value
        elif param.kind is param.VAR_KEYWORD:
            kwargs_dict.update(value)
    return args_list, kwargs_dict


def wrap_with_logging(
    func: Callable[..., Any], config: dict | None = None
) -> Callable[..., Any]:
    if getattr(func, "_menace_logging_wrapped", False):
        return func

    merged_config: dict[str, Any] = dict(_DEFAULT_WRAP_CONFIG)
    if config:
        merged_config.update(config)

    logger = get_logger(str(merged_config["logger_name"]))
    log_level = _resolve_level(merged_config["log_level"])
    include_args = bool(merged_config["include_args"])
    include_return = bool(merged_config["include_return"])
    include_exceptions = bool(merged_config["include_exceptions"])
    max_items = int(merged_config["max_items"])
    max_string_length = int(merged_config["max_string_length"])
    event_prefix = str(merged_config["log_event_prefix"])

    signature = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.monotonic()
        bound = signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        ordered_args, ordered_kwargs = _split_arguments(signature, bound)
        payload: MutableMapping[str, Any] = {
            "event": f"{event_prefix}{func.__name__}",
            "function_name": func.__name__,
            "module": func.__module__,
        }
        if include_args:
            args_value, args_truncated = _sanitize_and_truncate(
                ordered_args, max_items=max_items, max_string_length=max_string_length
            )
            kwargs_value, kwargs_truncated = _sanitize_and_truncate(
                ordered_kwargs, max_items=max_items, max_string_length=max_string_length
            )
            payload["args"] = args_value
            payload["kwargs"] = kwargs_value
            if args_truncated:
                payload["args_truncated"] = True
                payload["args_truncated_count"] = args_truncated
            if kwargs_truncated:
                payload["kwargs_truncated"] = True
                payload["kwargs_truncated_count"] = kwargs_truncated
        try:
            result = func(*args, **kwargs)
        except Exception as exc:
            duration_ms = (time.monotonic() - start) * 1000
            payload["duration_ms"] = duration_ms
            if include_exceptions:
                exception_value, exception_truncated = _sanitize_and_truncate(
                    {
                        "type": exc.__class__.__name__,
                        "message": str(exc),
                    },
                    max_items=max_items,
                    max_string_length=max_string_length,
                )
                payload["exception"] = exception_value
                payload["traceback"] = "".join(tb.format_exception(type(exc), exc, exc.__traceback__))
                if exception_truncated:
                    payload["exception_truncated"] = True
                    payload["exception_truncated_count"] = exception_truncated
            logger.log(log_level, payload["event"], extra=log_record(**payload))
            raise
        duration_ms = (time.monotonic() - start) * 1000
        payload["duration_ms"] = duration_ms
        if include_return:
            return_value, return_truncated = _sanitize_and_truncate(
                result, max_items=max_items, max_string_length=max_string_length
            )
            payload["return_value"] = return_value
            if return_truncated:
                payload["return_truncated"] = True
                payload["return_truncated_count"] = return_truncated
        logger.log(log_level, payload["event"], extra=log_record(**payload))
        return result

    wrapper.__signature__ = signature
    wrapper._menace_logging_wrapped = True  # type: ignore[attr-defined]
    wrapper._menace_logging_original = func  # type: ignore[attr-defined]
    return wrapper


@dataclass
class StabilizationLoggingWrapper:
    """Standardized logging for stabilization events."""

    correlation_id: str
    source: str | None = None
    logger_name: str = "stabilization.pipeline"

    def __post_init__(self) -> None:
        set_correlation_id(self.correlation_id)
        self._logger = get_logger(self.logger_name)

    @classmethod
    def start(
        cls, *, correlation_id: str | None = None, source: str | None = None
    ) -> "StabilizationLoggingWrapper":
        cid = correlation_id or f"stabilize-{uuid.uuid4()}"
        return cls(correlation_id=cid, source=source)

    def log_normalization(
        self, *, raw_text: str, normalized_text: str, sanitized_text: str, error: str | None
    ) -> None:
        self._log(
            "stabilization.normalization",
            ok=error is None,
            raw_length=len(raw_text),
            normalized_length=len(normalized_text),
            sanitized_length=len(sanitized_text),
            had_error=error is not None,
        )

    def log_validation(
        self,
        *,
        ok: bool,
        reason: str | None,
        diagnostics_count: int,
        error_category: str | None = None,
    ) -> None:
        self._log(
            "stabilization.validation",
            ok=ok,
            reason=reason,
            diagnostics_count=diagnostics_count,
            error_category=error_category,
        )

    def log_handoff(
        self,
        *,
        ok: bool,
        error: str | None,
        reason: str | None,
        error_category: str | None = None,
    ) -> None:
        self._log(
            "stabilization.handoff",
            ok=ok,
            error=error,
            reason=reason,
            error_category=error_category,
        )

    def close(self) -> None:
        set_correlation_id(None)

    def _log(self, event: str, *, ok: bool | None, **fields: Any) -> None:
        payload = {
            "event": event,
            "timestamp": _utc_timestamp(),
            "ok": ok,
        }
        if self.source:
            payload["source"] = self.source
        payload.update(fields)
        self._logger.info(event, extra=log_record(**payload))


__all__ = ["StabilizationLoggingWrapper", "wrap_with_logging", "_DEFAULT_WRAP_CONFIG"]
