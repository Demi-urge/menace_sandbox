"""Context builder for assembling cross-database retrieval context.

This lightweight module exposes :class:`ContextBuilder` which relies on
``UniversalRetriever`` to pull relevant records from the Error, Bot, Workflow
and Code databases.  Results are summarised for token efficiency and include
basic success/ROI style metrics when available.

The public entry point :func:`ContextBuilder.build_context` accepts a free form
query string and returns a JSON-like mapping::

    {
        "errors": [...],
        "bots": [...],
        "workflows": [...],
        "code": [...]
    }

Each list contains dictionaries with at least an ``id`` and ``summary`` field
and may optionally contain a ``metric`` representing ROI or success related
values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from universal_retriever import ResultBundle, UniversalRetriever

# Database wrappers ---------------------------------------------------------
# They are lightweight SQLite backed helpers throughout the codebase.  Only
# the minimal interface required by ``UniversalRetriever`` (``encode_text``,
# ``search_by_vector`` and ``get_vector``) is used here which makes the class
# easy to test with simple stubs.
from error_bot import ErrorDB
from bot_database import BotDB
from task_handoff_bot import WorkflowDB
from code_database import CodeDB

# Optional memory manager provides a summarisation helper.  When unavailable a
# tiny fallback summariser is used.
try:  # pragma: no cover - optional dependency
    from menace_memory_manager import MenaceMemoryManager, _summarise_text
except Exception:  # pragma: no cover - fallback when full package missing
    MenaceMemoryManager = None  # type: ignore

    def _summarise_text(text: str, ratio: float = 0.3) -> str:
        """Very small fallback summariser used during testing."""

        text = text.strip().replace("\n", " ")
        if len(text) <= 120:
            return text
        return text[:117] + "..."


@dataclass
class RecordContext:
    """Container for an individual record included in the context."""

    id: Any
    summary: str
    metric: float | None = None

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - simple helper
        data = {"id": self.id, "summary": self.summary}
        if self.metric is not None:
            data["metric"] = self.metric
        return data


class ContextBuilder:
    """Build compact context blocks from multiple databases."""

    def __init__(
        self,
        error_db: ErrorDB | None = None,
        bot_db: BotDB | None = None,
        workflow_db: WorkflowDB | None = None,
        code_db: CodeDB | None = None,
        memory_manager: Optional[MenaceMemoryManager] = None,
    ) -> None:
        self.error_db = error_db or ErrorDB()
        self.bot_db = bot_db or BotDB()
        self.workflow_db = workflow_db or WorkflowDB()
        self.code_db = code_db or CodeDB()
        self.memory = memory_manager

        # ``UniversalRetriever`` performs the heavy lifting of vector search
        # across the configured databases.
        self.retriever = UniversalRetriever(
            bot_db=self.bot_db,
            workflow_db=self.workflow_db,
            error_db=self.error_db,
        )

        # ``CodeDB`` might not always implement the embedding interface so we
        # register it best-effort.
        try:  # pragma: no cover - defensive
            self.retriever.register_db("code", self.code_db, ("id", "cid"))
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _summarise(self, text: str) -> str:
        """Summarise ``text`` using the memory manager if available."""

        if self.memory and hasattr(self.memory, "_summarise_text"):
            try:
                return self.memory._summarise_text(text)  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - fall back on error
                pass
        return _summarise_text(text)

    # ------------------------------------------------------------------
    def _metric(self, origin: str, meta: Dict[str, Any]) -> float | None:
        """Extract a success/ROI style metric from ``meta``."""

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
        except Exception:  # pragma: no cover - safety
            return None
        return None

    # ------------------------------------------------------------------
    def _bundle_to_record(self, bundle: ResultBundle) -> RecordContext:
        """Convert a :class:`ResultBundle` into :class:`RecordContext`."""

        meta = bundle.metadata
        origin = bundle.origin_db
        text = ""
        if origin == "error":
            text = meta.get("message") or meta.get("description") or ""
        elif origin == "bot":
            text = meta.get("name") or meta.get("purpose") or ""
        elif origin == "workflow":
            text = meta.get("title") or meta.get("description") or ""
        elif origin == "code":
            text = meta.get("summary") or meta.get("code") or ""

        summary = self._summarise(str(text))
        metric = self._metric(origin, meta)
        return RecordContext(id=bundle.record_id, summary=summary, metric=metric)

    # ------------------------------------------------------------------
    def build_context(self, query: str, limit_per_type: int = 3) -> Dict[str, List[Dict[str, Any]]]:
        """Return a mapping of summarised records for ``query``.

        Parameters
        ----------
        query:
            Free form text describing the current task or error.
        limit_per_type:
            Maximum number of entries to return for each record type.
        """

        # Fetch a generous number of results to allow fair distribution across
        # the different record types, then trim down per bucket.
        hits = self.retriever.retrieve(query, top_k=limit_per_type * 4)

        context: Dict[str, List[Dict[str, Any]]] = {
            "errors": [],
            "bots": [],
            "workflows": [],
            "code": [],
        }

        key_map = {
            "error": "errors",
            "bot": "bots",
            "workflow": "workflows",
            "code": "code",
        }

        for bundle in hits:
            key = key_map.get(bundle.origin_db)
            if not key:
                continue
            bucket = context[key]
            if len(bucket) >= limit_per_type:
                continue
            bucket.append(self._bundle_to_record(bundle).to_dict())

        return context


__all__ = ["ContextBuilder"]

