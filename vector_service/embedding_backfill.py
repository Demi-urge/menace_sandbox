from __future__ import annotations

"""Utilities for backfilling vector embeddings across databases."""

import logging
import time
from typing import List, Sequence
import importlib
import asyncio
from pathlib import Path

from .decorators import log_and_measure
from compliance.license_fingerprint import (
    check as license_check,
    fingerprint as license_fingerprint,
)

def _log_violation(path: str, lic: str, hash_: str) -> None:
    try:  # pragma: no cover - best effort
        CodeDB = importlib.import_module("code_database").CodeDB
        CodeDB().log_license_violation(path, lic, hash_)
    except Exception:
        pass

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
_RUN_SKIPPED = _me.Gauge(
    "embedding_backfill_skipped_total",
    "Records skipped during EmbeddingBackfill due to licensing",
    labelnames=["db", "license"],
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
    def _load_known_dbs(self, names: List[str] | None = None) -> List[type]:
        """Import all ``EmbeddableDBMixin`` subclasses dynamically.

        The repository is scanned for Python modules referencing
        :class:`EmbeddableDBMixin`.  Any classes found to inherit from the mixin
        are returned.  When ``names`` is provided the result is filtered to
        include only classes whose name matches any entry. Matching is
        case-insensitive and ignores plural forms or a trailing ``DB`` suffix.
        """

        root = Path(__file__).resolve().parents[1]
        for path in root.rglob("*.py"):  # pragma: no cover - best effort
            if any(part in {"tests", "scripts", "docs"} for part in path.parts):
                continue
            try:
                if "EmbeddableDBMixin" not in path.read_text(encoding="utf-8"):
                    continue
            except Exception:
                continue
            module = ".".join(path.relative_to(root).with_suffix("").parts)
            try:
                importlib.import_module(module)
            except Exception:
                continue
        try:
            subclasses = [
                cls
                for cls in EmbeddableDBMixin.__subclasses__()
                if hasattr(cls, "backfill_embeddings")
            ]
        except Exception:  # pragma: no cover - defensive
            subclasses = []
        if names:
            keys = [n.lower().rstrip("s") for n in names]
            filtered: List[type] = []
            for cls in subclasses:
                name = cls.__name__.lower()
                base = name[:-2] if name.endswith("db") else name
                for key in keys:
                    if name.startswith(key) or base.startswith(key):
                        filtered.append(cls)
                        break
            subclasses = filtered
        return subclasses

    # ------------------------------------------------------------------
    @log_and_measure
    def _process_db(
        self,
        db: EmbeddableDBMixin,
        *,
        batch_size: int,
        session_id: str = "",
    ) -> List[tuple[str, str]]:
        original_add = getattr(db, "add_embedding", None)
        skipped: List[tuple[str, str]] = []

        if callable(original_add):
            def wrapped_add(record_id, record, kind, *, source_id=""):
                text = record if isinstance(record, str) else str(record)
                lic = license_check(text)
                if lic:
                    _log_violation(
                        str(record_id),
                        lic,
                        license_fingerprint(text),
                    )
                    _RUN_SKIPPED.labels(db.__class__.__name__, lic).inc()
                    skipped.append((str(record_id), lic))
                    return
                return original_add(record_id, record, kind, source_id=source_id)

            db.add_embedding = wrapped_add  # type: ignore[attr-defined]

        try:
            db.backfill_embeddings(batch_size=batch_size)  # type: ignore[call-arg]
        except TypeError:
            db.backfill_embeddings()  # type: ignore[call-arg]
        return skipped

    # ------------------------------------------------------------------
    @log_and_measure
    def run(
        self,
        *,
        session_id: str = "",
        batch_size: int | None = None,
        backend: str | None = None,
        db: str | None = None,
        dbs: List[str] | None = None,
    ) -> None:
        """Backfill embeddings for ``EmbeddableDBMixin`` subclasses.

        If ``db`` or ``dbs`` is provided, only classes whose name matches any
        of the supplied values are processed. Matching is case-insensitive and
        ignores plural forms or a trailing ``DB`` suffix.
        """
        start = time.time()
        status = "success"
        try:
            bs = batch_size if batch_size is not None else self.batch_size
            be = backend or self.backend
            names = dbs or ([db] if db else None)
            subclasses = self._load_known_dbs(names=names)
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
                    skipped = self._process_db(db, batch_size=bs, session_id=session_id)
                    if skipped:
                        for rid, lic in skipped:
                            logger.warning(
                                "skipped %s due to license %s",
                                rid,
                                lic,
                                extra={"session_id": session_id},
                            )
                except Exception:  # pragma: no cover - best effort
                    continue
        except Exception:
            status = "failure"
            _RUN_OUTCOME.labels(status).inc()
            _RUN_DURATION.set(time.time() - start)
            raise
        _RUN_OUTCOME.labels(status).inc()
        _RUN_DURATION.set(time.time() - start)


async def schedule_backfill(
    *,
    batch_size: int | None = None,
    backend: str | None = None,
    dbs: Sequence[str] | None = None,
) -> None:
    """Asynchronously run :meth:`EmbeddingBackfill.run` for known databases.

    A single :class:`EmbeddingBackfill` instance is created and its
    :meth:`run` method is executed concurrently for each discovered
    :class:`EmbeddableDBMixin` subclass.  ``dbs`` can restrict execution to a
    subset of database names.
    """

    backfill = EmbeddingBackfill()
    if batch_size is not None:
        backfill.batch_size = batch_size
    if backend is not None:
        backfill.backend = backend

    subclasses = list(EmbeddableDBMixin.__subclasses__())
    if dbs:
        wanted = {d.lower() for d in dbs}
        subclasses = [c for c in subclasses if c.__name__.lower() in wanted]

    async def _run(cls: type) -> None:
        await asyncio.to_thread(backfill.run, db=cls.__name__)

    await asyncio.gather(*[_run(cls) for cls in subclasses])


__all__ = ["EmbeddingBackfill", "EmbeddableDBMixin", "schedule_backfill"]

