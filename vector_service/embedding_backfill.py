from __future__ import annotations

"""Utilities for backfilling vector embeddings across databases."""

import logging
import time
from typing import List

from .decorators import log_and_measure

try:  # pragma: no cover - optional dependency for metrics
    from . import metrics_exporter as _me  # type: ignore
except Exception:  # pragma: no cover - fallback when running standalone
    import metrics_exporter as _me  # type: ignore

_RUN_OUTCOME = _me.Gauge(
    "embedding_backfill_runs_total",
    "Outcomes of EmbeddingBackfill.run calls",
    labelnames=["status"],
)
_RUN_DURATION = _me.Gauge(
    "embedding_backfill_run_duration_seconds",
    "Duration of EmbeddingBackfill.run calls",
)

try:  # pragma: no cover - optional dependency
    from embeddable_db_mixin import EmbeddableDBMixin  # type: ignore
except Exception:  # pragma: no cover
    EmbeddableDBMixin = object  # type: ignore


class EmbeddingBackfill:
    """Trigger embedding backfills on all known database classes."""

    def __init__(self, batch_size: int = 100, backend: str = "annoy") -> None:
        self.batch_size = batch_size
        self.backend = backend

    # ------------------------------------------------------------------
    def _load_known_dbs(self) -> List[type]:
        """Import common database modules and return discovered classes."""

        modules = [
            "bot_database",
            "task_handoff_bot",
            "error_bot",
            "information_db",
            "chatgpt_enhancement_bot",
            "research_aggregator_bot",
        ]
        for name in modules:  # pragma: no cover - best effort imports
            try:
                __import__(name)
            except Exception:
                pass
        try:
            subclasses = [
                cls for cls in EmbeddableDBMixin.__subclasses__()
                if hasattr(cls, "backfill_embeddings")
            ]
        except Exception:  # pragma: no cover - defensive
            subclasses = []
        return subclasses

    # ------------------------------------------------------------------
    @log_and_measure
    def _process_db(
        self,
        db: EmbeddableDBMixin,
        *,
        batch_size: int,
        session_id: str = "",
    ) -> None:
        try:
            db.backfill_embeddings(batch_size=batch_size)  # type: ignore[call-arg]
        except TypeError:
            db.backfill_embeddings()  # type: ignore[call-arg]

    # ------------------------------------------------------------------
    @log_and_measure
    def run(
        self,
        *,
        session_id: str = "",
        batch_size: int | None = None,
        backend: str | None = None,
    ) -> None:
        """Backfill embeddings for all ``EmbeddableDBMixin`` subclasses."""
        start = time.time()
        status = "success"
        try:
            bs = batch_size if batch_size is not None else self.batch_size
            be = backend or self.backend
            subclasses = self._load_known_dbs()
            logger = logging.getLogger(__name__)
            total = len(subclasses)
            for idx, cls in enumerate(subclasses, 1):
                try:
                    db = cls(vector_backend=be)  # type: ignore[call-arg]
                except Exception:  # pragma: no cover - fallback
                    try:
                        db = cls()  # type: ignore[call-arg]
                    except Exception:
                        continue
                logger.info(
                    "Backfilling %s (%d/%d)",
                    cls.__name__,
                    idx,
                    total,
                    extra={"session_id": session_id},
                )
                try:
                    self._process_db(db, batch_size=bs, session_id=session_id)
                except Exception:  # pragma: no cover - best effort
                    continue
        except Exception:
            status = "failure"
            _RUN_OUTCOME.labels(status).inc()
            _RUN_DURATION.set(time.time() - start)
            raise
        _RUN_OUTCOME.labels(status).inc()
        _RUN_DURATION.set(time.time() - start)


__all__ = ["EmbeddingBackfill", "EmbeddableDBMixin"]

