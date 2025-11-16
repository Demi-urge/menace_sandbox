"""Utilities for lazily constructing shared :class:`DataBot` instances."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, Iterable, cast

from ..data_interfaces import DataBotInterface, RawMetrics

try:  # pragma: no cover - optional dependency during bootstrap
    from ..data_bot import DataBot as _RuntimeDataBot
except Exception as exc:  # pragma: no cover - optional dependency
    _DATA_BOT_IMPORT_ERROR: Exception | None = exc
    _RuntimeDataBot: type[DataBotInterface] | None = None
else:  # pragma: no cover - import succeeded
    _DATA_BOT_IMPORT_ERROR = None
    _RuntimeDataBot = cast("type[DataBotInterface]", _RuntimeDataBot)

_data_bot_fallback_logged = False
_shared_data_bot_instance: DataBotInterface | None = None


def _build_fallback_data_bot() -> DataBotInterface:
    """Return a minimal :class:`DataBotInterface` implementation."""

    class _FallbackDataBot:
        """Lightweight stand-in used when the real DataBot cannot load."""

        def __init__(self) -> None:
            self.db = SimpleNamespace(fetch=lambda *args, **kwargs: [])

        def collect(
            self,
            bot: str,
            response_time: float = 0.0,
            errors: int = 0,
            **metrics: float,
        ) -> RawMetrics:
            required = {"cpu", "memory", "disk_io", "net_io"}
            optional_fields = RawMetrics.__dataclass_fields__.keys()
            filtered_metrics: dict[str, Any] = {
                key: metrics[key]
                for key in optional_fields
                if key in metrics
                and key not in required
                and key not in {"bot", "errors", "response_time"}
            }
            return RawMetrics(
                bot=bot,
                cpu=float(metrics.get("cpu", 0.0)),
                memory=float(metrics.get("memory", 0.0)),
                response_time=response_time,
                disk_io=float(metrics.get("disk_io", 0.0)),
                net_io=float(metrics.get("net_io", 0.0)),
                errors=int(errors),
                **filtered_metrics,
            )

        def detect_anomalies(
            self,
            data: Any,
            metric: str,
            *,
            threshold: float = 3.0,
            metrics_db: Any | None = None,
        ) -> Iterable[int]:
            return []

        def roi(self, bot: str) -> float:
            return 0.0

    return cast(DataBotInterface, _FallbackDataBot())


def create_data_bot(logger: logging.Logger) -> DataBotInterface:
    """Instantiate the runtime data bot with graceful degradation."""

    global _data_bot_fallback_logged
    if _RuntimeDataBot is not None:
        try:
            return cast(DataBotInterface, _RuntimeDataBot())
        except Exception as exc:  # pragma: no cover - degraded bootstrap
            logger.warning(
                "DataBot initialisation failed for ModelAutomationPipeline: %s",
                exc,
            )
            _data_bot_fallback_logged = True
    else:
        if not _data_bot_fallback_logged:
            logger.warning(
                "DataBot unavailable for ModelAutomationPipeline: %s",
                _DATA_BOT_IMPORT_ERROR,
            )
            _data_bot_fallback_logged = True
    return _build_fallback_data_bot()


def get_data_bot(logger: logging.Logger | None = None) -> DataBotInterface:
    """Return a cached :class:`DataBotInterface` instance."""

    global _shared_data_bot_instance

    if _shared_data_bot_instance is None:
        logger = logger or logging.getLogger(__name__)
        _shared_data_bot_instance = create_data_bot(logger)
    return _shared_data_bot_instance


__all__ = ["create_data_bot", "get_data_bot"]
