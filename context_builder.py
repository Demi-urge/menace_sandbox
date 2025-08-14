from __future__ import annotations

"""Utilities for assembling retrieval context across internal databases.

This module exposes :class:`ContextBuilder` which queries a collection of
embeddable databases using :class:`UniversalRetriever`.  Results are distilled
into a compact JSON-like mapping optimised for downstream LLM consumption.

The builder emphasises high return on investment (ROI) or historically
successful outcomes when ranking items.  Existing ROI related fields on the
underlying records are used to weight scores where available.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

from universal_retriever import ResultBundle, UniversalRetriever

# Database imports -------------------------------------------------------------
# They live in separate modules but are lightweight wrappers over sqlite3
from bot_database import BotDB
from error_bot import ErrorDB
from task_handoff_bot import WorkflowDB
from failure_learning_system import DiscrepancyDB
from code_database import CodeDB


def _to_query_text(meta: Dict[str, Any]) -> str:
    """Flatten metadata values into a single query string."""

    parts: List[str] = []
    for val in meta.values():
        if isinstance(val, (str, int, float)):
            parts.append(str(val))
        elif isinstance(val, Sequence):
            parts.extend(str(v) for v in val if isinstance(v, (str, int, float)))
    return " ".join(parts)


def _snippet(text: str, length: int = 80) -> str:
    """Return a shortened preview of ``text`` for token efficiency."""

    text = text.strip().replace("\n", " ")
    return text if len(text) <= length else text[: length - 3] + "..."


@dataclass
class ContextItem:
    """Lightweight representation of a retrieved record."""

    id: Any
    snippet: str
    note: str
    roi: float | None = None

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        data = {"id": self.id, "snippet": self.snippet, "note": self.note}
        if self.roi is not None:
            data["roi"] = self.roi
        return data


class ContextBuilder:
    """Assemble compact context from heterogeneous data stores.

    Parameters
    ----------
    bot_db, workflow_db, error_db, discrepancy_db, code_db:
        Optional pre-initialised database instances.  When ``None`` the
        respective database is created with default parameters.
    """

    def __init__(
        self,
        *,
        bot_db: BotDB | None = None,
        workflow_db: WorkflowDB | None = None,
        error_db: ErrorDB | None = None,
        discrepancy_db: DiscrepancyDB | None = None,
        code_db: CodeDB | None = None,
    ) -> None:
        self.bot_db = bot_db or BotDB()
        self.workflow_db = workflow_db or WorkflowDB()
        self.error_db = error_db or ErrorDB()
        self.discrepancy_db = discrepancy_db or DiscrepancyDB()
        self.code_db = code_db or CodeDB()

        self.retriever = UniversalRetriever(
            bot_db=self.bot_db,
            workflow_db=self.workflow_db,
            error_db=self.error_db,
        )

        # Register CodeDB with the retriever when it exposes the required
        # vector search hooks.  ``CodeDB`` does not always ship with embedding
        # capabilities so this registration is best-effort.
        if all(
            hasattr(self.code_db, attr)
            for attr in ("search_by_vector", "encode_text", "get_vector")
        ):
            try:
                self.retriever.register_db("code", self.code_db, ("id", "cid"))
            except Exception:  # pragma: no cover - defensive
                pass

    # ------------------------------------------------------------------
    def _ranked_items(self, hits: Iterable[ResultBundle]) -> Dict[str, List[ContextItem]]:
        """Partition and ROI-rank retrieval ``hits`` by origin database."""

        buckets: Dict[str, List[ContextItem]] = {"error": [], "bot": [], "workflow": [], "code": []}

        for hit in hits:
            meta = hit.metadata
            origin = hit.origin_db
            roi_val: float | None = None
            if origin == "bot":
                roi_val = float(meta.get("estimated_profit", meta.get("roi", 0.0)) or 0.0)
                snippet = meta.get("name") or meta.get("purpose") or "bot"
            elif origin == "workflow":
                roi_val = float(meta.get("estimated_profit_per_bot", meta.get("roi", 0.0)) or 0.0)
                snippet = meta.get("title") or meta.get("description") or "workflow"
            elif origin == "error":
                snippet = meta.get("message") or meta.get("description") or "error"
            elif origin == "code":
                snippet = meta.get("summary") or meta.get("code", "code")
            else:
                snippet = str(meta)
            item = ContextItem(id=hit.record_id, snippet=_snippet(str(snippet)), note=hit.reason, roi=roi_val)
            if origin in buckets:
                buckets[origin].append(item)

        # Sort by ROI when available, otherwise by score already sorted in hits
        for key in ("bot", "workflow"):
            buckets[key].sort(key=lambda i: i.roi or 0.0, reverse=True)
        return buckets

    # ------------------------------------------------------------------
    def _discrepancy_items(self, query: str, limit: int = 5) -> List[ContextItem]:
        """Retrieve discrepancy detections matching ``query``."""

        items: List[ContextItem] = []
        try:
            cur = self.discrepancy_db.conn.execute(
                "SELECT rowid as id, rule, message, severity FROM detections ORDER BY severity DESC LIMIT ?",
                (limit,),
            )
            rows = cur.fetchall()
        except Exception:  # pragma: no cover - best effort
            rows = []
        for row in rows:
            note = f"{row['rule']} severity={row['severity']}"
            items.append(ContextItem(id=row["id"], snippet=_snippet(row["message"]), note=note, roi=None))
        return items

    # ------------------------------------------------------------------
    def build_context(self, metadata: Dict[str, Any], *, top_k: int = 5) -> Dict[str, Any]:
        """Return a compact context block derived from ``metadata``."""

        query = _to_query_text(metadata)
        hits = self.retriever.retrieve(query, top_k=top_k)
        buckets = self._ranked_items(hits)

        context = {
            "errors": [i.to_dict() for i in buckets["error"]],
            "bots": [i.to_dict() for i in buckets["bot"][:top_k]],
            "workflows": [i.to_dict() for i in buckets["workflow"][:top_k]],
            "code": [i.to_dict() for i in buckets["code"][:top_k]],
            "discrepancies": [i.to_dict() for i in self._discrepancy_items(query, limit=top_k)],
        }
        return context


__all__ = ["ContextBuilder"]
