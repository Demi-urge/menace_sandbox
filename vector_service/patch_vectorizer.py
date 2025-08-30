from __future__ import annotations

"""Embeds patch history records using textual fields."""

from pathlib import Path
from typing import Any, Dict, Iterator, Tuple, List

from embeddable_db_mixin import EmbeddableDBMixin
from code_database import PatchHistoryDB


class PatchVectorizer(EmbeddableDBMixin):
    """Embed patches by concatenating description, diff and summary."""

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
        self.db = PatchHistoryDB(path)
        self.conn = self.db.router.get_connection("patch_history")
        if index_path is None:
            index_path = Path(self.db.path).with_suffix(".patch.index")
        metadata_path = Path(index_path).with_suffix(".json")
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
