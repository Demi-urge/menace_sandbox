"""Compact cross‑database context builder.

This module exposes :class:`ContextBuilder` which gathers relevant records
from the light‑weight SQLite databases used throughout the project.  It wraps
``universal_retriever.UniversalRetriever`` and converts the returned
``ResultBundle`` objects into a small JSON block ready to drop into language
model prompts.  ROI or success metrics in the metadata are used to prefer
high–value records and an optional in‑memory cache avoids repeated work.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from universal_retriever import ResultBundle, UniversalRetriever

from .config import ContextBuilderConfig

# Database wrappers ---------------------------------------------------------
try:
    from .error_bot import ErrorDB  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    ErrorDB = object  # type: ignore
try:
    from .bot_database import BotDB  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    BotDB = object  # type: ignore
try:
    from .task_handoff_bot import WorkflowDB  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    WorkflowDB = object  # type: ignore
try:
    from .code_database import CodeDB  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    CodeDB = object  # type: ignore

try:  # Optional discrepancy database
    from failure_learning_system import DiscrepancyDB
except Exception:  # pragma: no cover - optional dependency
    DiscrepancyDB = None  # type: ignore

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


# Utility ------------------------------------------------------------------
def _resolve_db(obj: Any, cls: Any) -> Any:
    """Return a database instance given ``obj`` which may be a path or instance."""

    if obj is None:
        return cls()
    if isinstance(obj, (str, Path)):
        return cls(obj)
    return obj


@dataclass
class _ScoredEntry:
    entry: Dict[str, Any]
    score: float


class ContextBuilder:
    """Build compact JSON context blocks from multiple databases."""

    def __init__(
        self,
        *,
        bot_db: BotDB | str | Path | None = None,
        workflow_db: WorkflowDB | str | Path | None = None,
        error_db: ErrorDB | str | Path | None = None,
        code_db: CodeDB | str | Path | None = None,
        discrepancy_db: DiscrepancyDB | str | Path | None = None,
        memory_manager: Optional[MenaceMemoryManager] = None,
        db_weights: Dict[str, float] | None = None,
        max_tokens: int = ContextBuilderConfig().max_tokens,
    ) -> None:
        self.bot_db = _resolve_db(bot_db, BotDB)
        self.workflow_db = _resolve_db(workflow_db, WorkflowDB)
        self.error_db = _resolve_db(error_db, ErrorDB)
        self.code_db = _resolve_db(code_db, CodeDB)
        self.discrepancy_db = (
            _resolve_db(discrepancy_db, DiscrepancyDB)
            if discrepancy_db is not None and DiscrepancyDB is not None
            else None
        )

        self.memory = memory_manager
        self._cache: Dict[Tuple[str, int], str] = {}
        self.db_weights = db_weights or {}
        self.max_tokens = max_tokens

        # Assemble the universal retriever
        self.retriever = UniversalRetriever(
            bot_db=self.bot_db,
            workflow_db=self.workflow_db,
            error_db=self.error_db,
        )

        # Register optional databases on a best‑effort basis.
        try:  # pragma: no cover - defensive
            self.retriever.register_db("code", self.code_db, ("id", "cid"))
        except Exception:
            pass
        if self.discrepancy_db is not None:
            try:  # pragma: no cover - defensive
                self.retriever.register_db(
                    "discrepancy",
                    self.discrepancy_db,
                    ("id", "disc_id", "discrepancy_id"),
                )
            except Exception:
                pass

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
    def _bundle_to_entry(self, bundle: ResultBundle) -> Tuple[str, _ScoredEntry]:
        meta = bundle.metadata
        origin = bundle.origin_db

        text = ""
        entry: Dict[str, Any] = {"id": bundle.record_id}

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

        score = bundle.score + (metric or 0.0)
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

        hits = self.retriever.retrieve(query, top_k=top_k * 5)

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

