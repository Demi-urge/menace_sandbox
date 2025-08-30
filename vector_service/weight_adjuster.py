from __future__ import annotations

"""Adaptive per-vector weight adjustments for EmbeddableDBMixin databases."""

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence, Tuple
import asyncio

try:  # pragma: no cover - optional dependency
    from embeddable_db_mixin import EmbeddableDBMixin  # type: ignore
except Exception:  # pragma: no cover - fallback when mixin unavailable
    EmbeddableDBMixin = None  # type: ignore


@dataclass
class WeightAdjuster:
    """Adjust ranking weights for stored vectors.

    Parameters
    ----------
    dbs:
        Mapping of origin identifier to the corresponding
        :class:`EmbeddableDBMixin` instance holding the vectors.
    success_delta:
        Amount by which to increase weights for successful/high-ROI patches.
    failure_delta:
        Amount by which to decrease weights for failed or noisy patches.
    """

    dbs: Mapping[str, "EmbeddableDBMixin"]
    success_delta: float = 0.1
    failure_delta: float = 0.1

    # ------------------------------------------------------------------
    def adjust(
        self,
        vector_ids: Iterable[str | Tuple[str, str]],
        enhancement_score: float | None,
        roi_tag: str | None,
    ) -> None:
        """Update weights for ``vector_ids`` based on patch outcome."""

        score = float(enhancement_score or 0.0)
        success = self._is_positive(score, roi_tag)
        delta = (self.success_delta if success else -self.failure_delta) * (score or 1.0)
        touched: set[EmbeddableDBMixin] = set()

        for origin, rid in self._iter_ids(vector_ids):
            db = self.dbs.get(origin)
            if db is None:
                continue
            meta = getattr(db, "_metadata", {}).setdefault(rid, {})  # type: ignore[attr-defined]
            weight = float(meta.get("weight", 1.0))
            meta["weight"] = max(0.0, weight + delta)
            touched.add(db)

        for db in touched:
            try:
                db.save_index()  # type: ignore[attr-defined]
            except Exception:
                pass

    # ------------------------------------------------------------------
    async def bulk_adjust(
        self,
        updates: Sequence[Tuple[Iterable[str | Tuple[str, str]], float | None, str | None]],
    ) -> None:
        """Asynchronous helper to process multiple adjustments."""

        for ids, score, tag in updates:
            self.adjust(ids, score, tag)
            await asyncio.sleep(0)

    # ------------------------------------------------------------------
    @staticmethod
    def _iter_ids(vector_ids: Iterable[str | Tuple[str, str]]) -> Iterable[Tuple[str, str]]:
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
        negatives = ("low", "fail", "noise", "bug", "regret")
        positives = ("high", "success", "pass")
        if any(n in tag for n in negatives):
            return False
        if any(p in tag for p in positives):
            return True
        return enhancement_score >= 0.5


__all__ = ["WeightAdjuster"]
