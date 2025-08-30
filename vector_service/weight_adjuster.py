from __future__ import annotations

"""Adaptive ranking weight adjustments based on patch outcomes."""

from dataclasses import dataclass
from typing import Iterable, Tuple, Dict

from vector_metrics_db import VectorMetricsDB


@dataclass
class WeightAdjuster:
    """Adjust per-database ranking weights using patch metrics.

    Parameters
    ----------
    vector_metrics:
        Optional :class:`vector_metrics_db.VectorMetricsDB` instance used to
        persist weight updates.
    success_delta:
        Base amount by which to increase weights for positive patches.
    failure_delta:
        Base amount by which to decrease weights for noisy or failing patches.
    """

    vector_metrics: VectorMetricsDB | None = None
    success_delta: float = 0.1
    failure_delta: float = 0.1

    def __post_init__(self) -> None:  # pragma: no cover - best effort init
        if self.vector_metrics is None:
            try:
                self.vector_metrics = VectorMetricsDB()
            except Exception:
                self.vector_metrics = None

    # ------------------------------------------------------------------
    def adjust(
        self,
        vector_ids: Iterable[str | Tuple[str, str] | Tuple[str, str, float]],
        enhancement_score: float | None,
        roi_tag: str | None,
    ) -> Dict[str, float]:
        """Update ranking weights for origins and individual vectors.

        Returns a mapping of origin database to its new weight after the
        adjustment.  When neither ``enhancement_score`` nor ``roi_tag`` is
        provided no adjustments are made.  ``vector_ids`` may contain plain
        ``"origin:vector"`` strings, ``(origin, vector_id)`` tuples or
        ``(origin, vector_id, score)`` tuples.  When a score is provided it is
        used to scale the per-vector adjustment.
        """

        if self.vector_metrics is None:
            return {}
        if (enhancement_score in (None, 0.0)) and not roi_tag:
            return {}

        entries = list(self._iter_ids(vector_ids))
        origins = {origin for origin, _, _ in entries if origin}
        score = float(enhancement_score or 0.0)
        success = self._is_positive(score, roi_tag)
        base = self.success_delta if success else -self.failure_delta
        db_delta = base * (score or 1.0)

        # update individual vector weights
        for origin, rid, vscore in entries:
            key = f"{origin}:{rid}" if origin else rid
            delta = base * (vscore or 1.0)
            try:
                self.vector_metrics.update_vector_weight(key, delta)
            except Exception:
                pass

        updates: Dict[str, float] = {}
        for origin in origins:
            try:
                weight = self.vector_metrics.update_db_weight(origin, db_delta)
                updates[origin] = weight
                try:
                    self.vector_metrics.log_ranker_update(
                        origin, delta=db_delta, weight=weight
                    )
                except Exception:
                    pass
            except Exception:
                pass
        return updates

    # ------------------------------------------------------------------
    @staticmethod
    def _iter_ids(
        vector_ids: Iterable[str | Tuple[str, str] | Tuple[str, str, float]]
    ) -> Iterable[Tuple[str, str, float]]:
        for item in vector_ids:
            origin: str
            rid: str
            score: float = 1.0
            if isinstance(item, tuple):
                if len(item) >= 3:
                    origin, rid, score = item[0], item[1], float(item[2])
                elif len(item) == 2:
                    origin, rid = item
                else:
                    origin, rid = item[0], item[1] if len(item) > 1 else ""
            else:
                text = str(item)
                parts = text.split(":")
                if len(parts) >= 2:
                    origin, rid = parts[0], parts[1]
                    if len(parts) > 2:
                        try:
                            score = float(parts[2])
                        except Exception:
                            score = 1.0
                else:
                    origin, rid = "", text
            yield str(origin or ""), str(rid), float(score or 1.0)

    @staticmethod
    def _is_positive(enhancement_score: float, roi_tag: str | None) -> bool:
        tag = (roi_tag or "").lower()
        negatives = ("low", "fail", "noise", "bug")
        positives = ("high", "success", "pass")
        if any(n in tag for n in negatives):
            return False
        if any(p in tag for p in positives):
            return True
        return enhancement_score >= 0.5


__all__ = ["WeightAdjuster"]
