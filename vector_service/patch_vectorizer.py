from __future__ import annotations

"""Embeds patch history records using textual fields."""

import os
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple, List
import threading
from types import SimpleNamespace
import logging
import time

try:
    from menace_sandbox.embeddable_db_mixin import EmbeddableDBMixin
except ModuleNotFoundError:  # pragma: no cover - legacy flat import support
    from embeddable_db_mixin import EmbeddableDBMixin
from code_database import PatchHistoryDB
from dynamic_path_router import resolve_path


logger = logging.getLogger(__name__)


class PatchVectorizer(EmbeddableDBMixin):
    """Embed patches by concatenating description, diff and summary."""

    DB_FILE = "patch_history.db"

    DB_MODULE = "vector_service.patch_vectorizer"
    DB_CLASS = "PatchVectorizer"

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        index_path: str | Path | None = None,
        backend: str = "annoy",
        embedding_version: int = 1,
        bootstrap_fast: bool | None = None,
    ) -> None:
        init_start = time.perf_counter()
        warmup_context = any(
            os.getenv(flag, "").strip().lower() in {"1", "true", "yes", "on"}
            for flag in (
                "MENACE_BOOTSTRAP",
                "MENACE_BOOTSTRAP_FAST",
                "MENACE_BOOTSTRAP_MODE",
                "VECTOR_SERVICE_WARMUP",
                "VECTOR_SERVICE_LAZY_BOOTSTRAP",
            )
        )
        self._warmup_context = warmup_context
        if bootstrap_fast is None:
            bootstrap_fast = warmup_context
        logger.debug(
            "patch_vectorizer.init.start",
            extra={
                "path": str(path) if path is not None else None,
                "index_path": str(index_path) if index_path is not None else None,
                "backend": backend,
                "embedding_version": embedding_version,
                "bootstrap_fast": bootstrap_fast,
            },
        )
        self.bootstrap_fast = bool(bootstrap_fast)
        self._index_load_deferred = bool(self.bootstrap_fast or warmup_context)
        self._bootstrap_warmup = bool(self.bootstrap_fast or warmup_context)
        self._init_start = init_start
        self._activation_lock = threading.Lock()
        self._activation_thread: threading.Thread | None = None
        self._activation_started = False
        self._bootstrap_deferral_reason: str | None = None
        self._bootstrap_deferral_scheduled = False
        self._bootstrap_deferral_budget: float | None = None
        self._deferred_init: Dict[str, Any] = {
            "path": path,
            "index_path": index_path,
            "backend": backend,
            "embedding_version": embedding_version,
        }
        if self._bootstrap_warmup:
            self._prepare_warmup_stub(path, index_path, backend, embedding_version)
            return
        self._initialise_full(
            path=path,
            index_path=index_path,
            backend=backend,
            embedding_version=embedding_version,
            bootstrap_fast=self.bootstrap_fast,
            init_start=init_start,
        )

    # ------------------------------------------------------------------
    def _prepare_warmup_stub(
        self,
        path: str | Path | None,
        index_path: str | Path | None,
        backend: str,
        embedding_version: int,
    ) -> None:
        """Initialise a lightweight placeholder during bootstrap warmup."""

        self.db = SimpleNamespace(
            path=path if path is not None else self.DB_FILE,
            router=None,
            get=lambda *_a, **_k: None,
            _vec_db_enabled=False,
        )
        self.conn = None
        self.index_path = index_path if index_path is not None else f"{self.db.path}.patch.index"
        self.metadata_path = f"{self.index_path}.json" if self.index_path else None
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.embedding_version = embedding_version
        self.backend = backend
        self._model = None
        self._index = None
        self._vector_dim = 0
        self._id_map = []
        self._metadata = {}
        self._last_embedding_tokens = 0
        self._last_embedding_time = 0.0
        self._last_chunk_meta: Dict[str, Any] = {}
        logger.info(
            "patch_vectorizer.bootstrap_fast.stubbed",
            extra={
                "db_path": str(self.db.path),
                "index_path": str(self.index_path),
                "metadata_path": str(self.metadata_path),
                "warmup_context": self._warmup_context,
            },
        )

    def _activate_full_initialisation(self) -> None:
        """Transition from warmup placeholders to the real backing stores."""

        if not self._bootstrap_warmup:
            return

        with self._activation_lock:
            if not self._bootstrap_warmup:
                return
            if self._activation_started:
                return
            self._activation_started = True

        activation_start = time.perf_counter()
        deferred = dict(self._deferred_init)
        logger.info(
            "patch_vectorizer.bootstrap_fast.activate",
            extra={
                "db_path": str(deferred.get("path") or self.DB_FILE),
                "index_path": str(deferred.get("index_path") or ""),
                "warmup_context": self._warmup_context,
            },
        )
        self._bootstrap_warmup = False
        self.bootstrap_fast = False
        self._initialise_full(
            path=deferred.get("path"),
            index_path=deferred.get("index_path"),
            backend=deferred.get("backend", "annoy"),
            embedding_version=int(deferred.get("embedding_version", 1)),
            bootstrap_fast=False,
            init_start=activation_start,
        )
        self._bootstrap_deferral_reason = None
        self._bootstrap_deferral_scheduled = False
        self._bootstrap_deferral_budget = None

    def activate_async(self, *, reason: str | None = None) -> threading.Thread | None:
        """Hydrate the real index in the background when budgets permit."""

        if not self._bootstrap_warmup:
            return None

        with self._activation_lock:
            if self._activation_thread and self._activation_thread.is_alive():
                return self._activation_thread

            def _runner():
                logger.info(
                    "patch_vectorizer.bootstrap_fast.async_activation.start",
                    extra={"reason": reason, "warmup_context": self._warmup_context},
                )
                try:
                    self._activate_full_initialisation()
                    logger.info(
                        "patch_vectorizer.bootstrap_fast.async_activation.complete",
                        extra={"reason": reason, "warmup_context": self._warmup_context},
                    )
                except Exception:
                    logger.exception("patch_vectorizer async activation failed")

            thread = threading.Thread(target=_runner, daemon=True)
            thread.start()
            self._activation_thread = thread
            return thread

    def _bootstrap_budget_remaining(self) -> float | None:
        budget_env = os.getenv("MENACE_BOOTSTRAP_BUDGET_REMAINING")
        if budget_env:
            try:
                return float(budget_env)
            except ValueError:
                logger.debug("invalid MENACE_BOOTSTRAP_BUDGET_REMAINING value: %s", budget_env)
        return None

    def _bootstrap_background_executor(self) -> Any | None:
        try:
            from vector_service.lazy_bootstrap import _background_executor
        except Exception:
            return None

        try:
            return _background_executor()
        except Exception:  # pragma: no cover - advisory
            logger.debug("patch_vectorizer failed to acquire background executor", exc_info=True)
            return None

    def _record_bootstrap_deferral(
        self,
        *,
        reason: str,
        budget_remaining: float | None,
        scheduled: bool,
    ) -> None:
        self._bootstrap_deferral_reason = reason
        self._bootstrap_deferral_scheduled = scheduled
        self._bootstrap_deferral_budget = budget_remaining
        logger.info(
            "patch_vectorizer.bootstrap_fast.deferred",
            extra={
                "reason": reason,
                "budget_remaining": budget_remaining,
                "scheduled": scheduled,
                "warmup_context": self._warmup_context,
            },
        )
        try:
            from vector_service import lazy_bootstrap

            update_cache = getattr(lazy_bootstrap, "_update_warmup_stage_cache", None)
            if callable(update_cache):
                update_cache(
                    "patch_vectorizer",
                    "deferred",
                    logger,
                    meta={
                        "reason": reason,
                        "scheduled": scheduled,
                        "budget_remaining": budget_remaining,
                    },
                    emit_metric=False,
                )
        except Exception:  # pragma: no cover - best effort metadata
            logger.debug("failed to record patch_vectorizer warmup deferral", exc_info=True)

    def _background_activation_hook(self, *, reason: str) -> None:
        budget_remaining = self._bootstrap_budget_remaining()
        if budget_remaining is not None and budget_remaining <= 0:
            self._record_bootstrap_deferral(
                reason=reason, budget_remaining=budget_remaining, scheduled=False
            )
            return

        scheduled = False
        executor = self._bootstrap_background_executor()
        if executor is not None:
            try:
                executor.submit(self._activate_full_initialisation)
                scheduled = True
            except Exception:  # pragma: no cover - background best effort
                logger.debug("background executor scheduling failed", exc_info=True)

        if not scheduled:
            scheduled = self.activate_async(reason=reason) is not None

        self._record_bootstrap_deferral(
            reason=reason, budget_remaining=budget_remaining, scheduled=scheduled
        )

    def _gate_bootstrap_request(self, *, reason: str) -> bool:
        if not self._bootstrap_warmup:
            return False
        self._background_activation_hook(reason=reason)
        return True

    def _initialise_full(
        self,
        *,
        path: str | Path | None,
        index_path: str | Path | None,
        backend: str,
        embedding_version: int,
        bootstrap_fast: bool,
        init_start: float,
    ) -> None:
        db_path: Path | str | None
        if path is not None:
            path_resolve_start = time.perf_counter()
            try:
                db_path = Path(resolve_path(str(path)))
            except FileNotFoundError:
                db_path = Path(path).resolve()
            logger.info(
                "patch_vectorizer.db_path.resolved path=%s duration=%.6fs",
                db_path,
                time.perf_counter() - path_resolve_start,
                extra={
                    "path": str(db_path),
                    "duration_s": round(time.perf_counter() - path_resolve_start, 6),
                },
            )
        else:
            db_path = None

        db_init_start = time.perf_counter()
        self.db = PatchHistoryDB(db_path, bootstrap_fast=bootstrap_fast)
        logger.info(
            "patch_vectorizer.db.init path=%s duration=%.6fs",
            self.db.path,
            time.perf_counter() - db_init_start,
            extra={
                "db_path": str(self.db.path),
                "duration_s": round(time.perf_counter() - db_init_start, 6),
            },
        )
        conn_start = time.perf_counter()
        self.conn = self.db.router.get_connection("patch_history")
        logger.debug(
            "patch_vectorizer.db.connection",
            extra={"duration_s": round(time.perf_counter() - conn_start, 6)},
        )

        try:
            base = Path(resolve_path(str(self.db.path)))
        except FileNotFoundError:
            base = Path(self.db.path).resolve()

        if index_path is None:
            index_candidate = base.with_suffix(".patch.index")
        else:
            index_candidate = Path(index_path)
        index_resolve_start = time.perf_counter()
        try:
            index_path = Path(resolve_path(str(index_candidate)))
        except FileNotFoundError:
            index_path = index_candidate
        logger.info(
            "patch_vectorizer.index.resolved path=%s duration=%.6fs",
            index_path,
            time.perf_counter() - index_resolve_start,
            extra={
                "index_path": str(index_path),
                "duration_s": round(time.perf_counter() - index_resolve_start, 6),
            },
        )

        metadata_candidate = Path(index_path).with_suffix(".json")
        metadata_resolve_start = time.perf_counter()
        try:
            metadata_path = Path(resolve_path(str(metadata_candidate)))
        except FileNotFoundError:
            metadata_path = metadata_candidate
        logger.debug(
            "patch_vectorizer.metadata.resolved path=%s duration=%.6fs",
            metadata_path,
            time.perf_counter() - metadata_resolve_start,
            extra={
                "metadata_path": str(metadata_path),
                "duration_s": round(time.perf_counter() - metadata_resolve_start, 6),
            },
        )
        EmbeddableDBMixin.__init__(
            self,
            index_path=index_path,
            metadata_path=metadata_path,
            backend=backend,
            embedding_version=embedding_version,
            defer_index_load=self._index_load_deferred,
        )
        logger.info(
            "patch_vectorizer.init.complete duration=%.6fs index=%s",
            time.perf_counter() - init_start,
            index_path,
            extra={
                "duration_s": round(time.perf_counter() - init_start, 6),
                "index_path": str(index_path),
                "index_load_deferred": self._index_load_deferred,
            },
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _compose(record: Any) -> str:
        """Return text for embedding from ``record``.

        ``record`` may be a mapping or an object with ``description``, ``diff``
        and ``summary`` attributes.  Missing fields default to empty strings.
        """

        if isinstance(record, dict):
            desc = record.get("description") or ""
            diff = record.get("diff") or ""
            summary = record.get("summary") or ""
        else:
            desc = getattr(record, "description", "") or ""
            diff = getattr(record, "diff", "") or ""
            summary = getattr(record, "summary", "") or ""
        return "\n".join(part for part in (desc, diff, summary) if part)

    def transform(self, record: Dict[str, Any]) -> List[float]:
        """Return embedding for ``record``."""
        return self.encode_text(self._compose(record))

    # ``EmbeddableDBMixin`` expects ``vector`` and ``iter_records``
    vector = transform

    def iter_records(self) -> Iterator[Tuple[int, Dict[str, Any], str]]:
        if self._gate_bootstrap_request(reason="iter_records"):
            return iter(())
        cur = self.conn.execute("SELECT id FROM patch_history")
        for (pid,) in cur.fetchall():
            rec = self.db.get(pid)
            if rec is None:
                continue
            yield pid, {
                "description": getattr(rec, "description", None),
                "diff": getattr(rec, "diff", None),
                "summary": getattr(rec, "summary", None),
            }, "patch"

    def _ensure_index_loaded(self) -> None:
        if self._gate_bootstrap_request(reason="index_load"):
            return
        if self._index_loaded:
            return
        start = time.perf_counter()
        logger.info(
            "patch_vectorizer.index.load.start",
            extra={
                "deferred": self._index_load_deferred,
                "index_path": str(self.index_path),
                "metadata_path": str(self.metadata_path),
            },
        )
        super()._ensure_index_loaded()
        logger.info(
            "patch_vectorizer.index.load.complete",
            extra={
                "deferred": self._index_load_deferred,
                "duration_s": round(time.perf_counter() - start, 6),
                "index_path": str(self.index_path),
                "metadata_path": str(self.metadata_path),
            },
        )


def backfill_patch_embeddings(
    path: str | Path | None = None,
    *,
    index_path: str | Path | None = None,
    backend: str = "annoy",
    embedding_version: int = 1,
) -> None:
    """Backfill embeddings for existing patch history records."""

    pv = PatchVectorizer(
        path=path,
        index_path=index_path,
        backend=backend,
        embedding_version=embedding_version,
    )
    pv.backfill_embeddings()


__all__ = ["PatchVectorizer", "backfill_patch_embeddings"]
