"""Compact cross‑database context builder.

This module exposes :class:`ContextBuilder` which gathers relevant records
from the light‑weight SQLite databases used throughout the project.  It relies
on :class:`semantic_service.Retriever` to gather relevant records and converts
the returned dictionaries into a small JSON block ready to drop into language
model prompts.  ROI or success metrics in the metadata are used to prefer
high–value records and an optional in‑memory cache avoids repeated work.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from semantic_service import Retriever

from .config import ContextBuilderConfig

# Optional summariser -------------------------------------------------------
try:  # pragma: no cover - heavy dependency
    from menace_memory_manager import MenaceMemoryManager, _summarise_text
except Exception:  # pragma: no cover - tiny fallback helper
    MenaceMemoryManager = None  # type: ignore

    def _summarise_text(text: str, ratio: float = 0.3) -> str:
        text = text.strip().replace("\n", " ")
        if len(text) <= 120:
            return text
        return text[:117] + "..."


@dataclass
class _ScoredEntry:
    entry: Dict[str, Any]
    score: float


class ContextBuilder:
    """Build compact JSON context blocks from multiple databases."""

    def __init__(
        self,
        *,
        retriever: Retriever | None = None,
        memory_manager: Optional[MenaceMemoryManager] = None,
        db_weights: Dict[str, float] | None = None,
        max_tokens: int = ContextBuilderConfig().max_tokens,
    ) -> None:
        self.retriever = retriever or Retriever()
        self.memory = memory_manager
        self._cache: Dict[Tuple[str, int], str] = {}
        self.db_weights = db_weights or {}
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------
    def _summarise(self, text: str) -> str:
        if self.memory and hasattr(self.memory, "_summarise_text"):
            try:
                return self.memory._summarise_text(text)  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - fallback
                pass
        return _summarise_text(text)

    # ------------------------------------------------------------------
    def _metric(self, origin: str, meta: Dict[str, Any]) -> float | None:
        """Extract ROI/success metrics from metadata."""

        try:
            if origin == "error":
                freq = meta.get("frequency")
                if freq is not None:
                    return 1.0 / (1.0 + float(freq))

            if origin == "bot":
                for key in ("roi", "deploy_count"):
                    if key in meta and meta[key] is not None:
                        return float(meta[key])

            if origin == "workflow":
                for key in ("roi", "usage", "runs"):
                    if key in meta and meta[key] is not None:
                        return float(meta[key])

            if origin == "code":
                for key in ("roi", "patch_success"):
                    if key in meta and meta[key] is not None:
                        return float(meta[key])

            if origin == "discrepancy":
                for key in ("roi", "severity", "impact"):
                    if key in meta and meta[key] is not None:
                        return float(meta[key])
        except Exception:  # pragma: no cover - defensive
            return None
        return None

    # ------------------------------------------------------------------
    def _bundle_to_entry(self, bundle: Dict[str, Any]) -> Tuple[str, _ScoredEntry]:
        meta = bundle.get("metadata", {})
        origin = bundle.get("origin_db", "")

        text = ""
        entry: Dict[str, Any] = {"id": bundle.get("record_id")}

        if origin == "error":
            text = meta.get("message") or meta.get("description") or ""
        elif origin == "bot":
            text = meta.get("name") or meta.get("purpose") or ""
            if "name" in meta:
                entry["name"] = meta["name"]
        elif origin == "workflow":
            text = meta.get("title") or meta.get("description") or ""
            if "title" in meta:
                entry["title"] = meta["title"]
        elif origin == "discrepancy":
            text = meta.get("message") or meta.get("description") or ""
        elif origin == "code":
            text = meta.get("summary") or meta.get("code") or ""

        entry["desc"] = self._summarise(str(text))
        metric = self._metric(origin, meta)
        if metric is not None:
            entry["metric"] = metric

        score = float(bundle.get("score", 0.0)) + (metric or 0.0)
        score *= self.db_weights.get(origin, 1.0)

        key_map = {
            "error": "errors",
            "bot": "bots",
            "workflow": "workflows",
            "code": "code",
            "discrepancy": "discrepancies",
        }
        return key_map.get(origin, ""), _ScoredEntry(entry, score)

    # ------------------------------------------------------------------
    def build_context(self, query: str, top_k: int = 5, **_: Any) -> str:
        """Return a compact JSON context for ``query``.

        Results are retrieved from each database, prioritising entries with
        high ROI or success metrics.  ``top_k`` limits how many items from each
        source are included in the final JSON.  Repeated calls with the same
        ``query`` and ``top_k`` are served from an in‑memory cache.
        """

        cache_key = (query, top_k)
        if cache_key in self._cache:
            return self._cache[cache_key]

        hits = self.retriever.search(query, top_k=top_k * 5)

        buckets: Dict[str, List[_ScoredEntry]] = {
            "errors": [],
            "bots": [],
            "workflows": [],
            "code": [],
            "discrepancies": [],
        }

        for bundle in hits:
            key, scored = self._bundle_to_entry(bundle)
            if key:
                buckets[key].append(scored)

        result: Dict[str, List[Dict[str, Any]]] = {k: [] for k in buckets}
        for key, items in buckets.items():
            items.sort(key=lambda s: s.score, reverse=True)
            result[key] = [s.entry for s in items[:top_k]]

        # Enforce global token limit with a rough estimate (characters / 4)
        def _estimate(data: Dict[str, List[Dict[str, Any]]]) -> int:
            return len(
                json.dumps(data, ensure_ascii=False, separators=(",", ":"))
            ) // 4

        token_estimate = _estimate(result)
        if token_estimate > self.max_tokens:
            keys = list(result.keys())
            while token_estimate > self.max_tokens and any(result.values()):
                changed = False
                for key in keys:
                    if result[key]:
                        result[key].pop()
                        changed = True
                        token_estimate = _estimate(result)
                        if token_estimate <= self.max_tokens:
                            break
                if not changed:
                    break

        compact = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        self._cache[cache_key] = compact
        return compact


__all__ = ["ContextBuilder"]

