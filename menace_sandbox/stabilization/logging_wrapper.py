"""Logging helpers for stabilization pipeline events."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import uuid
from typing import Any

from logging_utils import get_logger, log_record, set_correlation_id


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


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


__all__ = ["StabilizationLoggingWrapper"]
