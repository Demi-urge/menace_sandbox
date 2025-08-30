from __future__ import annotations

"""Adjust per-vector ranking weights based on patch outcomes."""

from dataclasses import dataclass
from typing import Iterable, Mapping, Tuple

from vector_metrics_db import VectorMetricsDB


@dataclass
class WeightAdjuster:
    """Simple adaptive weighting for individual vectors.

    Parameters
    ----------
    vector_metrics:
        Optional :class:`vector_metrics_db.VectorMetricsDB` instance used to
        persist weight updates.  When omitted a default instance is created.
    success_delta:
        Amount by which to increase the weight of vectors associated with a
        successful patch.
    failure_delta:
        Amount by which to decrease the weight of vectors associated with a
        failed patch or negative ROI.
    """

    vector_metrics: VectorMetricsDB | None = None
    success_delta: float = 0.1
    failure_delta: float = 0.1

    def __post_init__(self) -> None:
        if self.vector_metrics is None:
            try:
                self.vector_metrics = VectorMetricsDB()
            except Exception:
                self.vector_metrics = None

    # ------------------------------------------------------------------
    def adjust(
        self,
        vectors: Iterable[Tuple[str, str, float]],
        success: bool,
        *,
        roi_deltas: Mapping[str, float] | None = None,
    ) -> None:
        """Update weights for *vectors* based on patch outcome."""

        if self.vector_metrics is None:
            return
        for origin, vid, _score in vectors:
            key = f"{origin}:{vid}" if origin else vid
            delta = self.success_delta if success else -self.failure_delta
            if roi_deltas is not None:
                try:
                    roi = float(roi_deltas.get(origin or "", 0.0))
                    if roi < 0:
                        delta = -self.failure_delta
                except Exception:
                    pass
            try:
                self.vector_metrics.update_vector_weight(key, delta)
            except Exception:
                pass
