from __future__ import annotations

"""Embeds patch history records using textual fields."""

from pathlib import Path
from typing import Any, Dict, Iterator, Tuple, List

try:
    from menace_sandbox.embeddable_db_mixin import EmbeddableDBMixin
except ModuleNotFoundError:  # pragma: no cover - legacy flat import support
    from embeddable_db_mixin import EmbeddableDBMixin
from code_database import PatchHistoryDB
from dynamic_path_router import resolve_path


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
    ) -> None:
        db_path: Path | str | None
        if path is not None:
            try:
                db_path = Path(resolve_path(str(path)))
            except FileNotFoundError:
                db_path = Path(path).resolve()
        else:
            db_path = None

        self.db = PatchHistoryDB(db_path)
        self.conn = self.db.router.get_connection("patch_history")

        try:
            base = Path(resolve_path(str(self.db.path)))
        except FileNotFoundError:
            base = Path(self.db.path).resolve()

        if index_path is None:
            index_candidate = base.with_suffix(".patch.index")
        else:
            index_candidate = Path(index_path)
        try:
            index_path = Path(resolve_path(str(index_candidate)))
        except FileNotFoundError:
            index_path = index_candidate

        metadata_candidate = Path(index_path).with_suffix(".json")
        try:
            metadata_path = Path(resolve_path(str(metadata_candidate)))
        except FileNotFoundError:
            metadata_path = metadata_candidate
        EmbeddableDBMixin.__init__(
            self,
            index_path=index_path,
            metadata_path=metadata_path,
            backend=backend,
            embedding_version=embedding_version,
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
