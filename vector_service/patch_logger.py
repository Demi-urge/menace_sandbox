from __future__ import annotations

"""Helper for recording patch outcomes for contributing vectors."""

from typing import Any, Iterable, List, Sequence, Tuple

from .decorators import log_and_measure

try:  # pragma: no cover - optional dependencies
    from vector_metrics_db import VectorMetricsDB  # type: ignore
except Exception:  # pragma: no cover
    VectorMetricsDB = None  # type: ignore

try:  # pragma: no cover
    from code_database import PatchHistoryDB  # type: ignore
except Exception:  # pragma: no cover
    PatchHistoryDB = None  # type: ignore


class PatchLogger:
    """Record patch outcomes in ``PatchHistoryDB`` or ``VectorMetricsDB``.

    ``metrics_db`` is accepted for backwards compatibility and maps to the
    older :class:`data_bot.MetricsDB` API.  When provided it takes precedence
    over the more granular databases.
    """

    def __init__(
        self,
        patch_db: PatchHistoryDB | None = None,
        vector_metrics: VectorMetricsDB | None = None,
        metrics_db: Any | None = None,
    ) -> None:
        self.patch_db = patch_db or (PatchHistoryDB() if PatchHistoryDB is not None else None)
        self.vector_metrics = vector_metrics or (
            VectorMetricsDB() if VectorMetricsDB is not None else None
        )
        self.metrics_db = metrics_db

    # ------------------------------------------------------------------
    def _parse_vectors(self, vector_ids: Iterable[str]) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        for vid in vector_ids:
            if ":" in vid:
                origin, vec_id = vid.split(":", 1)
            else:
                origin, vec_id = "", vid
            pairs.append((origin, vec_id))
        return pairs

    # ------------------------------------------------------------------
    @log_and_measure
    def track_contributors(
        self,
        vector_ids: Sequence[str],
        result: bool,
        *,
        patch_id: str = "",
        session_id: str = "",
    ) -> None:
        """Log patch outcome for vectors contributing to a patch."""

        pairs = self._parse_vectors(vector_ids)

        if self.metrics_db is not None:
            try:  # pragma: no cover - legacy path
                self.metrics_db.log_patch_outcome(
                    patch_id or "", result, pairs, session_id=session_id
                )
            except Exception:
                pass
        elif self.patch_db is not None and patch_id:
            try:  # pragma: no cover - best effort
                self.patch_db.record_vector_metrics(
                    session_id,
                    pairs,
                    patch_id=int(patch_id),
                    contribution=0.0,
                    win=result,
                    regret=not result,
                )
            except Exception:
                pass
        elif self.vector_metrics is not None:
            try:  # pragma: no cover - best effort
                self.vector_metrics.update_outcome(
                    session_id,
                    pairs,
                    contribution=0.0,
                    patch_id=patch_id,
                    win=result,
                    regret=not result,
                )
            except Exception:
                pass


__all__ = ["PatchLogger"]

