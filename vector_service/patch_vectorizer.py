from __future__ import annotations

"""Embeds patch history records using textual fields."""

from pathlib import Path
from typing import Any, Dict, Iterator, Tuple, List
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
        bootstrap_fast: bool = False,
    ) -> None:
        init_start = time.perf_counter()
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
        if self.bootstrap_fast:
            fast_db_path = Path(path) if path is not None else Path(self.DB_FILE)
            index_fallback = Path(index_path) if index_path is not None else fast_db_path.with_suffix(
                ".patch.index"
            )
            metadata_fallback = index_fallback.with_suffix(".json")
            stub_conn = SimpleNamespace(execute=lambda *_a, **_k: SimpleNamespace(fetchall=lambda: []))
            stub_router = SimpleNamespace(get_connection=lambda *_a, **_k: stub_conn)
            self.db = SimpleNamespace(
                path=fast_db_path,
                router=stub_router,
                get=lambda *_a, **_k: None,
                _vec_db_enabled=False,
            )
            self.conn = stub_router.get_connection("patch_history")
            self.index_path = index_fallback
            self.metadata_path = metadata_fallback
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
            self.encode_text = lambda _text: []  # type: ignore[assignment]
            logger.info(
                "patch_vectorizer.bootstrap_fast.stubbed",
                extra={
                    "db_path": str(fast_db_path),
                    "index_path": str(index_fallback),
                    "metadata_path": str(metadata_fallback),
                },
            )
            return
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
        )
        logger.info(
            "patch_vectorizer.init.complete duration=%.6fs index=%s",
            time.perf_counter() - init_start,
            index_path,
            extra={
                "duration_s": round(time.perf_counter() - init_start, 6),
                "index_path": str(index_path),
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
