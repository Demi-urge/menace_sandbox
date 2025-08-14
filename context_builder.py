"""Compact cross-database context builder.

This module exposes :class:`ContextBuilder` which aggregates relevant records
from a number of light‑weight SQLite backed databases.  It relies on the
existing :mod:`universal_retriever` for vector similarity and then ranks the
results using success or ROI style metrics when available.  The final context is
returned as a compact JSON string containing only ids, short descriptions and
useful metrics so that downstream consumers can remain within token budgets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from universal_retriever import ResultBundle, UniversalRetriever
try:  # pragma: no cover - support package and standalone usage
    from config import get_config  # type: ignore
except Exception:  # pragma: no cover - fallback to package import
    from menace.config import get_config  # type: ignore

# Database wrappers ---------------------------------------------------------
from error_bot import ErrorDB
from bot_database import BotDB
from task_handoff_bot import WorkflowDB
from code_database import CodeDB
from failure_learning_system import DiscrepancyDB

# Optional summariser -------------------------------------------------------
try:  # pragma: no cover - optional heavy dependency
    from menace_memory_manager import MenaceMemoryManager, _summarise_text
except Exception:  # pragma: no cover - fall back to tiny helper
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
class Record:
    id: Any
    desc: str
    metric: float | None
    score: float


class ContextBuilder:
    """Build compact JSON context blocks from multiple databases."""

    def __init__(
        self,
        *,
        error_db: ErrorDB | str | Path | None = None,
        bot_db: BotDB | str | Path | None = None,
        workflow_db: WorkflowDB | str | Path | None = None,
        discrepancy_db: DiscrepancyDB | str | Path | None = None,
        code_db: CodeDB | str | Path | None = None,
        memory_manager: Optional[MenaceMemoryManager] = None,
    ) -> None:
        self.error_db = _resolve_db(error_db, ErrorDB)
        self.bot_db = _resolve_db(bot_db, BotDB)
        self.workflow_db = _resolve_db(workflow_db, WorkflowDB)
        self.discrepancy_db = (
            _resolve_db(discrepancy_db, DiscrepancyDB)
            if discrepancy_db is not None
            else None
        )
        self.code_db = _resolve_db(code_db, CodeDB)
        self.memory = memory_manager

        self.retriever = UniversalRetriever(
            bot_db=self.bot_db,
            workflow_db=self.workflow_db,
            error_db=self.error_db,
        )

        # Optional databases may not implement the embedding interface; register
        # them on a best‑effort basis.
        try:  # pragma: no cover - defensive
            self.retriever.register_db("code", self.code_db, ("id", "cid"))
        except Exception:
            pass
        try:  # pragma: no cover - defensive
            self.retriever.register_db(
                "discrepancy", self.discrepancy_db, ("id", "disc_id", "discrepancy_id")
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
                if "resolution_success" in meta:
                    return float(meta["resolution_success"])
                freq = meta.get("frequency")
                if freq is not None:
                    return 1.0 / (1.0 + float(freq))

            if origin == "bot":
                for key in ("roi", "estimated_profit", "deploy_count"):
                    if key in meta and meta[key] is not None:
                        return float(meta[key])

            if origin == "workflow":
                for key in ("roi", "usage", "runs"):
                    if key in meta and meta[key] is not None:
                        return float(meta[key])

            if origin == "code":
                for key in ("patch_success", "coverage", "roi"):
                    if key in meta and meta[key] is not None:
                        return float(meta[key])

            if origin == "discrepancy":
                for key in ("severity", "roi", "impact"):
                    if key in meta and meta[key] is not None:
                        return float(meta[key])
        except Exception:  # pragma: no cover - defensive
            return None
        return None

    # ------------------------------------------------------------------
    def _bundle_to_record(self, bundle: ResultBundle, weight: float) -> Record:
        meta = bundle.metadata
        origin = bundle.origin_db

        text = ""
        if origin == "error":
            text = meta.get("message") or meta.get("description") or ""
        elif origin == "bot":
            text = meta.get("name") or meta.get("purpose") or ""
        elif origin == "workflow":
            text = meta.get("title") or meta.get("description") or ""
        elif origin == "discrepancy":
            text = meta.get("message") or meta.get("description") or ""
        elif origin == "code":
            text = meta.get("summary") or meta.get("code") or ""
            text = str(text)
            if len(text) > 60:
                text = text[:57] + "..."
            summary = text
            metric = self._metric(origin, meta)
            score = bundle.score + weight * (metric or 0.0)
            return Record(bundle.record_id, summary, metric, score)

        summary = self._summarise(str(text))
        metric = self._metric(origin, meta)
        score = bundle.score + weight * (metric or 0.0)
        return Record(bundle.record_id, summary, metric, score)

    # ------------------------------------------------------------------
    def build_context(
        self,
        task_desc: str,
        limit_per_db: int = 3,
        *,
        max_tokens: int | None = None,
        metric_weight: float = 1.0,
        db_weights: Dict[str, float] | None = None,
        **_: Any,
    ) -> str:
        """Return a compact JSON context for ``task_desc``.

        ``max_tokens`` is a soft limit; lower priority items are trimmed until
        the output roughly fits within the specified budget. When ``None`` the
        value is read from ``config.context_builder.max_tokens`` or defaults to
        800.
        ``metric_weight`` controls how strongly ROI/success metrics influence
        ranking relative to vector similarity. ``db_weights`` may bias metric
        contributions for individual databases.
        """

        cfg = None
        try:  # pragma: no cover - best effort
            cfg = get_config().context_builder  # type: ignore[attr-defined]
        except Exception:
            cfg = None

        if max_tokens is None:
            max_tokens = getattr(cfg, "max_tokens", 800)
        if db_weights is None:
            db_weights = dict(getattr(cfg, "db_weights", {}))

        hits = self.retriever.retrieve(task_desc, top_k=limit_per_db * 5)

        buckets: Dict[str, List[Record]] = {
            "errors": [],
            "bots": [],
            "workflows": [],
            "discrepancies": [],
            "code": [],
        }
        key_map = {
            "error": "errors",
            "bot": "bots",
            "workflow": "workflows",
            "discrepancy": "discrepancies",
            "code": "code",
        }

        for bundle in hits:
            key = key_map.get(bundle.origin_db)
            if not key:
                continue
            weight = metric_weight * db_weights.get(bundle.origin_db, 1.0)
            buckets[key].append(self._bundle_to_record(bundle, weight))

        all_items: List[tuple[str, Record]] = []
        for key, recs in buckets.items():
            recs.sort(key=lambda r: r.score, reverse=True)
            recs[:] = recs[:limit_per_db]
            all_items.extend((key, r) for r in recs)

        def build_dict(items: List[tuple[str, Record]]) -> Dict[str, List[Dict[str, Any]]]:
            result: Dict[str, List[Dict[str, Any]]] = {k: [] for k in buckets}
            for kind, r in items:
                entry = {"id": r.id, "desc": r.desc}
                if r.metric is not None:
                    entry["metric"] = r.metric
                result[kind].append(entry)
            return result

        def token_len(obj: Dict[str, Any]) -> int:
            return len(json.dumps(obj, ensure_ascii=False)) // 4

        context_dict = build_dict(all_items)
        while all_items and token_len(context_dict) > max_tokens:
            all_items.sort(key=lambda x: x[1].score)
            all_items.pop(0)
            context_dict = build_dict(all_items)

        return json.dumps(context_dict, ensure_ascii=False, separators=(",", ":"))


__all__ = ["ContextBuilder"]

