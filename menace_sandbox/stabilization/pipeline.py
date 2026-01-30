"""Stabilize and validate LLM outputs for downstream consumers."""

from __future__ import annotations

from typing import Any, Dict
import re

from completion_validator import validate_completion
from error_ontology import classify_exception

from .logging_wrapper import StabilizationLoggingWrapper

_CODE_FENCE_PATTERN = re.compile(r"^```(?:[a-zA-Z0-9_-]+)?\n(?P<body>.*)```$", re.DOTALL)
_TRACEBACK_PATTERN = re.compile(r"Traceback \(most recent call last\):.*", re.DOTALL)
_ERROR_LINE_PATTERN = re.compile(
    r"^\s*(?:[A-Za-z_]*Error|Exception|RuntimeError|ValueError|TypeError):.+",
    re.MULTILINE,
)


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    match = _CODE_FENCE_PATTERN.match(text)
    if match:
        return match.group("body").strip()
    return text


def _sanitize_errors(text: str) -> tuple[str, str | None]:
    error: str | None = None
    trace = _TRACEBACK_PATTERN.search(text)
    if trace:
        error = trace.group(0).strip()
        text = text.replace(trace.group(0), "").strip()
    if text and _ERROR_LINE_PATTERN.match(text):
        error = error or _ERROR_LINE_PATTERN.match(text).group(0).strip()
        text = ""
    return text, error


def _classify_error(error: str | None) -> str | None:
    if not error:
        return None
    try:
        return str(classify_exception(Exception(error), error))
    except Exception:
        return None


def _extract_raw_text(raw_output: Any) -> tuple[str, str | None]:
    if isinstance(raw_output, BaseException):
        return "", str(raw_output)
    if raw_output is None:
        return "", None
    if isinstance(raw_output, bytes):
        return raw_output.decode("utf-8", errors="replace"), None
    if isinstance(raw_output, dict):
        error = raw_output.get("error")
        if raw_output.get("text") is not None:
            return str(raw_output["text"]), str(error) if error else None
        if raw_output.get("content") is not None:
            return str(raw_output["content"]), str(error) if error else None
        return str(raw_output), str(error) if error else None
    if hasattr(raw_output, "text"):
        raw_text = str(getattr(raw_output, "text"))
        raw_meta = getattr(raw_output, "raw", None)
        error = None
        if isinstance(raw_meta, dict):
            error = raw_meta.get("error")
        return raw_text, str(error) if error else None
    return str(raw_output), None


def stabilize_completion(
    raw_output: Any, *, source: str | None = None, correlation_id: str | None = None
) -> Dict[str, Any]:
    """Normalize, sanitize, and validate a raw completion payload."""

    logger = StabilizationLoggingWrapper.start(
        correlation_id=correlation_id,
        source=source,
    )
    try:
        raw_text, raw_error = _extract_raw_text(raw_output)
        normalized = _strip_code_fences(raw_text)
        sanitized, sanitized_error = _sanitize_errors(normalized)
        error = sanitized_error or raw_error
        error_category = _classify_error(error)

        logger.log_normalization(
            raw_text=raw_text,
            normalized_text=normalized,
            sanitized_text=sanitized,
            error=error,
        )

        validation = validate_completion(sanitized)
        ok = validation.ok
        reason = validation.reason
        if not ok and error and not reason:
            reason = error

        logger.log_validation(
            ok=ok,
            reason=reason,
            diagnostics_count=len(validation.diagnostics),
            error_category=error_category,
        )

        payload: Dict[str, Any] = {
            "ok": ok,
            "text": validation.text if ok else "",
            "normalized_text": sanitized,
            "raw_text": raw_text,
            "error": error,
            "error_category": error_category,
            "reason": reason,
            "diagnostics": list(validation.diagnostics),
        }
        if source:
            payload["source"] = source

        logger.log_handoff(
            ok=ok,
            error=error,
            reason=reason,
            error_category=error_category,
        )
        return payload
    finally:
        logger.close()


__all__ = ["stabilize_completion"]
