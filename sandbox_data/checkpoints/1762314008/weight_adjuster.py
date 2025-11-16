from __future__ import annotations

"""Adaptive ranking weight adjustments based on patch outcomes."""

from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover - PyYAML missing
    yaml = None  # type: ignore

from vector_metrics_db import VectorMetricsDB
from .roi_tags import RoiTag
from dynamic_path_router import resolve_path


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

    try:
        cfg_path = resolve_path("config/roi_tag_sentiment.yaml")
    except FileNotFoundError:
        cfg_path = None

    if cfg_path and cfg_path.exists() and yaml is not None:
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
        vectors: Iterable[Tuple[str, str, float, RoiTag | str | None]],
        *,
        error_trace_count: int | None = None,
        tests_passed: bool | None = None,
    ) -> Dict[str, float]:
        """Update ranking weights using per-vector patch metrics.

        Parameters
        ----------
        vectors:
            Iterable of ``(origin, vector_id, enhancement_score, roi_tag)``
            tuples describing the vectors contributing to a patch.  ``origin``
            may be an empty string when unknown.  ``enhancement_score`` and
            ``roi_tag`` are used to scale the adjustment applied to the
            vector's ranking weight.
        error_trace_count:
            Number of error traces observed.  Higher counts reduce the
            magnitude of adjustments.
        tests_passed:
            Optional boolean indicating whether tests passed.  Failed tests
            invert the sign of the adjustment.
        """

        if self.vector_metrics is None:
            return {}

        entries: list[Tuple[str, str, float, RoiTag]] = []
        for origin, rid, enh, tag in vectors:
            entries.append(
                (
                    str(origin or ""),
                    str(rid),
                    float(enh or 0.0),
                    RoiTag.validate(tag),
                )
            )
        if not entries:
            return {}

        origins: Dict[str, float] = {}
        for origin, rid, enh, tag in entries:
            sentiment = self.tag_sentiment.get(tag, 1.0)
            factor = float(enh or 1.0) * sentiment
            if tests_passed is not None:
                factor *= 1.0 if tests_passed else -1.0
            if error_trace_count:
                factor /= 1.0 + float(error_trace_count)

            base = (
                self.vector_success_delta if factor >= 0 else -self.vector_failure_delta
            )
            delta = base * abs(factor)
            key = f"{origin}:{rid}" if origin else rid
            try:
                self.vector_metrics.update_vector_weight(key, delta)
            except Exception:
                pass

            if origin and origin not in origins:
                origins[origin] = factor

        updates: Dict[str, float] = {}
        for origin, factor in origins.items():
            db_base = self.db_success_delta if factor >= 0 else -self.db_failure_delta
            db_delta = db_base * abs(factor)
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

        

__all__ = ["WeightAdjuster"]
