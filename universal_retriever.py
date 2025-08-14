from __future__ import annotations

from typing import Any, Iterable, List, Sequence


class UniversalRetriever:
    """Search across multiple databases using vector similarity."""

    def __init__(
        self,
        *,
        bot_db: Any | None = None,
        workflow_db: Any | None = None,
        error_db: Any | None = None,
        enhancement_db: Any | None = None,
        information_db: Any | None = None,
    ) -> None:
        self.bot_db = bot_db
        self.workflow_db = workflow_db
        self.error_db = error_db
        self.enhancement_db = enhancement_db
        self.information_db = information_db

        self._encoder = next(
            (
                db
                for db in (
                    bot_db,
                    workflow_db,
                    error_db,
                    enhancement_db,
                    information_db,
                )
                if db is not None
            ),
            None,
        )
        if self._encoder is None:
            raise ValueError("At least one database instance is required")

    # ------------------------------------------------------------------
    def _extract_id(self, obj: Any, names: Sequence[str]) -> Any | None:
        if isinstance(obj, dict):
            for name in names:
                if name in obj:
                    return obj[name]
            return None
        for name in names:
            if hasattr(obj, name):
                return getattr(obj, name)
        return None

    def _vector_for_query(self, query: Any) -> List[float]:
        if isinstance(query, str):
            return self._encoder.encode_text(query)

        mapping = [
            (self.bot_db, ("id", "bid")),
            (self.workflow_db, ("id", "wid")),
            (self.error_db, ("id",)),
            (self.enhancement_db, ("id",)),
            (self.information_db, ("id", "info_id")),
        ]
        for db, names in mapping:
            if db is None:
                continue
            rec_id = self._extract_id(query, names)
            if rec_id is not None:
                vec = db.get_vector(rec_id)
                if vec is not None:
                    return vec
        raise TypeError("Unsupported query type for retrieval")

    # ------------------------------------------------------------------
    def retrieve(self, query: Any, top_k: int = 10) -> List[Any]:
        """Return top results from all databases for ``query``."""

        vector = self._vector_for_query(query)
        candidates: List[Any] = []
        for db in (
            self.bot_db,
            self.workflow_db,
            self.error_db,
            self.enhancement_db,
            self.information_db,
        ):
            if db is None:
                continue
            try:
                matches = db.search_by_vector(vector, top_k)
            except Exception:  # pragma: no cover - defensive
                continue
            candidates.extend(matches)

        def _dist(item: Any) -> float:
            if isinstance(item, dict):
                return float(item.get("_distance", float("inf")))
            return float(getattr(item, "_distance", float("inf")))

        candidates.sort(key=_dist)
        return candidates[:top_k]
