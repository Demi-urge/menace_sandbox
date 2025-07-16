from __future__ import annotations
"""Optional Sentry wrapper for error reporting."""

import logging

try:
    import sentry_sdk  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    sentry_sdk = None  # type: ignore


class SentryClient:
    """Lightweight wrapper around ``sentry_sdk``."""

    def __init__(self, dsn: str | None = None, *, fallback_logger: logging.Logger | None = None, **kwargs: object) -> None:
        self.logger = fallback_logger or logging.getLogger(__name__)
        if sentry_sdk:
            sentry_sdk.init(dsn=dsn, **kwargs)
            self.client = sentry_sdk
        else:  # pragma: no cover - fallback
            self.client = None

    def capture_exception(self, exc: Exception) -> None:
        if self.client:
            try:
                self.client.capture_exception(exc)
            except Exception as e:
                self.logger.error("failed to send exception to Sentry: %s", e)


__all__ = ["SentryClient"]
