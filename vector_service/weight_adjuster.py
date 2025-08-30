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
        vector_ids: Iterable[str | Tuple[str, str]],
        enhancement_score: float | None,
        roi_tag: str | None,
    ) -> Dict[str, float]:
        """Update ranking weights for origins present in ``vector_ids``.

        Returns a mapping of origin database to its new weight after the
        adjustment.  When neither ``enhancement_score`` nor ``roi_tag`` is
        provided no adjustments are made.
        """

        if self.vector_metrics is None:
            return {}
        if (enhancement_score in (None, 0.0)) and not roi_tag:
            return {}

        origins = {origin for origin, _ in self._iter_ids(vector_ids) if origin}
        score = float(enhancement_score or 0.0)
        success = self._is_positive(score, roi_tag)
        delta = (self.success_delta if success else -self.failure_delta) * (score or 1.0)

        updates: Dict[str, float] = {}
        for origin in origins:
            try:
                weight = self.vector_metrics.update_db_weight(origin, delta)
                updates[origin] = weight
                try:
                    self.vector_metrics.log_ranker_update(
                        origin, delta=delta, weight=weight
                    )
                except Exception:
                    pass
            except Exception:
                pass
        return updates

    # ------------------------------------------------------------------
    @staticmethod
    def _iter_ids(
        vector_ids: Iterable[str | Tuple[str, str]]
    ) -> Iterable[Tuple[str, str]]:
        for item in vector_ids:
            if isinstance(item, tuple):
                if len(item) == 2:
                    origin, rid = item
                else:
                    origin, rid = item[0], item[1]
            else:
                text = str(item)
                if ":" in text:
                    origin, rid = text.split(":", 1)
                else:
                    origin, rid = "", text
            yield str(origin or ""), str(rid)

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
