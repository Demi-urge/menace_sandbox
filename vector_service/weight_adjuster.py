from __future__ import annotations

"""Adaptive ranking weight adjustments based on patch outcomes."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Tuple

import yaml

from vector_metrics_db import VectorMetricsDB
from .roi_tags import RoiTag


def _load_tag_sentiment() -> Dict[RoiTag, float]:
    """Load ROI tag sentiment mapping from YAML config.

    The default mapping assigns ``1.0`` to positive tags and ``-1.0`` to
    negative ones.  ``config/roi_tag_sentiment.yaml`` can override these
    defaults with a simple ``tag: float`` mapping.
    """

    mapping: Dict[RoiTag, float] = {
        RoiTag.SUCCESS: 1.0,
        RoiTag.HIGH_ROI: 1.0,
        RoiTag.LOW_ROI: -1.0,
        RoiTag.BUG_INTRODUCED: -1.0,
        RoiTag.NEEDS_REVIEW: -1.0,
        RoiTag.BLOCKED: -1.0,
    }

    cfg_path = (
        Path(__file__).resolve().parent.parent / "config" / "roi_tag_sentiment.yaml"
    )
    if cfg_path.exists():
        try:  # pragma: no cover - configuration loading is best effort
            data = yaml.safe_load(cfg_path.read_text()) or {}
            if isinstance(data, dict):
                for key, val in data.items():
                    try:
                        tag = RoiTag(key)
                    except ValueError:
                        continue
                    try:
                        mapping[tag] = float(val)
                    except Exception:
                        continue
        except Exception:
            pass

    return mapping


ROI_TAG_SENTIMENT = _load_tag_sentiment()


@dataclass
class WeightAdjuster:
    """Adjust ranking weights for vector origins and individual vectors."""

    vector_metrics: VectorMetricsDB | None = None
    db_success_delta: float = 0.1
    db_failure_delta: float = 0.1
    vector_success_delta: float = 0.1
    vector_failure_delta: float = 0.1
    tag_sentiment: Dict[RoiTag, float] = field(
        default_factory=lambda: ROI_TAG_SENTIMENT.copy()
    )

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
        roi_tag: RoiTag | str | None,
        *,
        error_trace_count: int | None = None,
        tests_passed: bool | None = None,
    ) -> Dict[str, float]:
        """Update ranking weights using patch quality metrics.

        Parameters
        ----------
        vector_ids:
            Iterable containing ``origin:vector`` identifiers or tuples of
            ``(origin, vector, score)``.  ``score`` scales the per-vector
            adjustment.
        enhancement_score:
            Numeric patch enhancement score used to scale the base weight
            delta.
        roi_tag:
            ROI tag describing the patch outcome.  The tag's numeric sentiment
            controls the sign of the weight update.
        error_trace_count:
            Number of error traces observed.  Higher counts reduce the
            magnitude of adjustments.
        tests_passed:
            Optional boolean indicating whether tests passed.  Failed tests
            invert the sign of the adjustment.
        """

        if self.vector_metrics is None:
            return {}

        entries = list(self._iter_ids(vector_ids))
        if not entries:
            return {}

        origins = {origin for origin, _, _ in entries if origin}

        roi_tag_val = RoiTag.validate(roi_tag) if roi_tag is not None else None
        sentiment = self.tag_sentiment.get(roi_tag_val, 1.0)

        factor = float(enhancement_score or 1.0) * sentiment
        if tests_passed is not None:
            factor *= 1.0 if tests_passed else -1.0
        if error_trace_count:
            factor /= 1.0 + float(error_trace_count)

        # update individual vector weights
        for origin, rid, vscore in entries:
            vfactor = factor * (vscore or 1.0)
            base = (
                self.vector_success_delta if vfactor >= 0 else -self.vector_failure_delta
            )
            delta = base * abs(vfactor)
            key = f"{origin}:{rid}" if origin else rid
            try:
                self.vector_metrics.update_vector_weight(key, delta)
            except Exception:
                pass

        db_base = self.db_success_delta if factor >= 0 else -self.db_failure_delta
        db_delta = db_base * abs(factor)
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
        

__all__ = ["WeightAdjuster"]
