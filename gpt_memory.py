"""Simple persistent memory for GPT interactions.

This module exposes :class:`GPTMemoryManager` which stores prompt/response pairs
along with optional tags and timestamps.  Data is persisted using a tiny SQLite
database.  When the optional :mod:`sentence_transformers` package is available a
vector embedding is stored for each prompt allowing semantic search.

For backwards compatibility the module also exposes :class:`GPTMemory` – a thin
wrapper around the project's :class:`menace_memory_manager.MenaceMemoryManager`.
This wrapper is exercised in the unit tests and provides a minimal ``store`` and
``retrieve`` API.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence

_HELPER_NAME = "import_compat"
_PACKAGE_NAME = "menace_sandbox"

try:  # pragma: no cover - prefer package import when installed
    from menace_sandbox import import_compat  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - support flat execution
    _helper_path = Path(__file__).resolve().parent / f"{_HELPER_NAME}.py"
    _spec = importlib.util.spec_from_file_location(
        f"{_PACKAGE_NAME}.{_HELPER_NAME}",
        _helper_path,
    )
    if _spec is None or _spec.loader is None:  # pragma: no cover - defensive
        raise
    import_compat = importlib.util.module_from_spec(_spec)
    sys.modules[f"{_PACKAGE_NAME}.{_HELPER_NAME}"] = import_compat
    sys.modules[_HELPER_NAME] = import_compat
    _spec.loader.exec_module(import_compat)
else:  # pragma: no cover - ensure helper aliases exist
    sys.modules.setdefault(_HELPER_NAME, import_compat)
    sys.modules.setdefault(f"{_PACKAGE_NAME}.{_HELPER_NAME}", import_compat)

import_compat.bootstrap(__name__, __file__)
load_internal = import_compat.load_internal

_db_router = load_internal("db_router")
DBRouter = _db_router.DBRouter
init_db_router = _db_router.init_db_router

GPTMemoryInterface = load_internal("gpt_memory_interface").GPTMemoryInterface

log_embedding_metrics = load_internal("embeddable_db_mixin").log_embedding_metrics

find_semantic_risks = load_internal("analysis.semantic_diff_filter").find_semantic_risks

govern_retrieval = load_internal("governed_retrieval").govern_retrieval

SharedVectorService = load_internal("vector_service").SharedVectorService

redact_secrets = load_internal("security.secret_redactor").redact


try:  # Optional dependency used for event publication
    UnifiedEventBus = load_internal("unified_event_bus").UnifiedEventBus
except ModuleNotFoundError:  # pragma: no cover - optional dependency missing
    UnifiedEventBus = None  # type: ignore
except Exception:  # pragma: no cover - degrade gracefully
    UnifiedEventBus = None  # type: ignore

try:  # Optional dependency for graph updates
    KnowledgeGraph = load_internal("knowledge_graph").KnowledgeGraph
except ModuleNotFoundError:  # pragma: no cover - optional dependency missing
    KnowledgeGraph = None  # type: ignore
except Exception:  # pragma: no cover - degrade gracefully
    KnowledgeGraph = None  # type: ignore

try:  # Optional dependency used for semantic embeddings
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - keep import lightweight
    SentenceTransformer = None  # type: ignore

try:  # Optional dependency used by the light wrapper ``GPTMemory``
    _memory_module = load_internal("menace_memory_manager")
except ModuleNotFoundError:  # pragma: no cover - tests stub this module
    MenaceMemoryManager = None  # type: ignore

    def _summarise_text(text: str, ratio: float = 0.2) -> str:  # pragma: no cover - fallback
        """Fallback summariser used when menace_memory_manager is unavailable."""

        return text[: max(1, int(len(text) * ratio))]
except Exception:  # pragma: no cover - degrade gracefully in tests
    MenaceMemoryManager = None  # type: ignore

    def _summarise_text(text: str, ratio: float = 0.2) -> str:  # pragma: no cover - fallback
        """Fallback summariser used when menace_memory_manager is unavailable."""

        return text[: max(1, int(len(text) * ratio))]
else:
    MenaceMemoryManager = _memory_module.MenaceMemoryManager  # type: ignore[attr-defined]
    _summarise_text = _memory_module._summarise_text  # type: ignore[attr-defined]

# --------------------------------------------------------------------------- tags
try:  # Canonical tag constants shared across modules
    _log_tags = load_internal("log_tags")
except ModuleNotFoundError:  # pragma: no cover - fallback when tags unavailable
    FEEDBACK = "feedback"
    IMPROVEMENT_PATH = "improvement_path"
    ERROR_FIX = "error_fix"
    INSIGHT = "insight"
except Exception:  # pragma: no cover - degrade gracefully
    FEEDBACK = "feedback"
    IMPROVEMENT_PATH = "improvement_path"
    ERROR_FIX = "error_fix"
    INSIGHT = "insight"
else:
    FEEDBACK = _log_tags.FEEDBACK
    IMPROVEMENT_PATH = _log_tags.IMPROVEMENT_PATH
    ERROR_FIX = _log_tags.ERROR_FIX
    INSIGHT = _log_tags.INSIGHT

# Standardised tag set for GPT interaction logging
STANDARD_TAGS = {FEEDBACK, IMPROVEMENT_PATH, ERROR_FIX, INSIGHT}

logger = logging.getLogger(__name__)


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the cosine similarity between two vectors."""

    import math

    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    denom = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / denom if denom else 0.0


@dataclass
class MemoryEntry:
    """Representation of a stored interaction returned by ``search_context``."""

    prompt: str
    response: str
    tags: List[str]
    timestamp: str
    score: float = 0.0
    metadata: Dict[str, Any] | None = None


class GPTMemoryManager(GPTMemoryInterface):
    """Persist and query GPT interactions using SQLite.

    Parameters
    ----------
    db_path:
        Location of the SQLite database.  ``"gpt_memory.db"`` by default.
    embedder:
        Optional :class:`SentenceTransformer` instance.  When provided each
        prompt is embedded and semantic search can be performed.  Supplying an
        embedder allows callers to share a pre-initialised model across
        components rather than constructing one internally.
    event_bus:
        Optional :class:`UnifiedEventBus`.  When supplied, each call to
        :meth:`log_interaction` publishes a ``"memory:new"`` event containing
        the interaction details.
    knowledge_graph:
        Optional :class:`KnowledgeGraph` instance used to mirror interactions into
        the shared knowledge graph.
    """

    def __init__(
        self,
        db_path: str | Path = "gpt_memory.db",
        *,
        embedder: SentenceTransformer | None = None,
        event_bus: Optional[UnifiedEventBus] = None,
        knowledge_graph: "KnowledgeGraph | None" = None,
        vector_service: SharedVectorService | None = None,
        router: "DBRouter | None" = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.router = router or _db_router.GLOBAL_ROUTER
        if self.router is None:
            self.router = init_db_router("memory", str(self.db_path), str(self.db_path))
        self.conn = self.router.get_connection("memory")
        self.embedder = embedder
        self.event_bus = event_bus
        self.graph = knowledge_graph
        # ``vector_service`` takes precedence when supplied.  Otherwise we lazily
        # construct a service only if an ``embedder`` is provided, avoiding
        # initialising a service that cannot embed text.
        self.vector_service = vector_service
        if self.vector_service is None and embedder is not None:
            self.vector_service = SharedVectorService(embedder)
        self._ensure_schema()

    # ------------------------------------------------------------------ utils
    def _ensure_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                tags TEXT,
                ts TEXT NOT NULL,
                embedding TEXT,
                alerts TEXT
            )
            """
        )
        cols = [r[1] for r in self.conn.execute("PRAGMA table_info(interactions)").fetchall()]
        if "alerts" not in cols:
            self.conn.execute("ALTER TABLE interactions ADD COLUMN alerts TEXT")
        self.conn.commit()

    # --------------------------------------------------------------- interface
    def log_interaction(
        self,
        prompt: str,
        response: str,
        tags: Sequence[str] | None = None,
    ) -> None:
        """Record a GPT interaction in persistent storage."""
        original_prompt = prompt
        prompt = redact_secrets(prompt)
        if prompt != original_prompt:
            logger.warning("redacted secrets in prompt before embedding")
        timestamp = datetime.utcnow().isoformat()
        tag_list = list(tags or [])
        tag_str = ",".join(tag_list)
        # Avoid storing duplicate prompt/response pairs
        cur = self.conn.execute(
            "SELECT 1 FROM interactions WHERE prompt = ? AND response = ? LIMIT 1",
            (prompt, response),
        )
        if cur.fetchone() is not None:
            return
        alerts = find_semantic_risks(original_prompt.splitlines())
        if alerts:
            logger.warning(
                "semantic risks detected: %s", [a[1] for a in alerts]
            )
        embedding: str | None = None
        tokens = 0
        wall_time = 0.0
        if self.vector_service is not None and not alerts:
            try:
                start = perf_counter()
                vec = self.vector_service.vectorise_and_store(
                    "text", timestamp, {"text": original_prompt}
                )
                wall_time = perf_counter() - start
                tokenizer = getattr(self.vector_service.text_embedder, "tokenizer", None)
                if tokenizer:
                    tokens = len(tokenizer.encode(prompt))
                embedding = json.dumps([float(x) for x in vec])
            except Exception:  # pragma: no cover - embedding is optional
                embedding = None
                tokens = 0
                wall_time = 0.0

        store_start = perf_counter()
        cur = self.conn.execute(
            "INSERT INTO interactions(prompt, response, tags, ts, embedding, alerts)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                prompt,
                response,
                tag_str,
                timestamp,
                embedding,
                json.dumps(alerts) if alerts else None,
            ),
        )
        self.conn.commit()
        store_latency = perf_counter() - store_start

        if embedding is not None:
            vector_id = str(cur.lastrowid)
            log_embedding_metrics(
                self.__class__.__name__, tokens, wall_time, store_latency, vector_id=vector_id
            )

        if self.event_bus:
            try:
                self.event_bus.publish(
                    "memory:new", {"prompt": prompt, "tags": tag_list}
                )
            except Exception:  # pragma: no cover - defensive
                pass

        if self.graph:
            try:
                self.graph.add_memory_entry(prompt, tag_list)
                bots = [t.split(":", 1)[1] for t in tag_list if t.startswith("bot:")]
                codes = [t.split(":", 1)[1] for t in tag_list if t.startswith("code:")]
                errs = [
                    t.split(":", 1)[1]
                    for t in tag_list
                    if t.startswith("error:") or t.startswith("error_category:")
                ]
                if bots or codes or errs:
                    self.graph.add_gpt_insight(
                        prompt,
                        bots=bots or None,
                        code_paths=codes or None,
                        error_categories=errs or None,
                    )
            except Exception:  # pragma: no cover - defensive
                pass

    def search_context(
        self,
        query: str,
        *,
        limit: int = 5,
        tags: Sequence[str] | None = None,
        use_embeddings: bool = False,
    ) -> List[MemoryEntry]:
        """Search stored interactions matching ``query``.

        When ``use_embeddings`` is true and a vector service is available cosine
        similarity between the query and stored prompts is used; otherwise a
        simple substring search over prompt/response is performed.
        """

        redacted_query = redact_secrets(query)
        if redacted_query != query:
            logger.warning("redacted secrets in query before embedding")
        params: list[Any] = []
        where: list[str] = []
        if tags:
            for t in tags:
                where.append("tags LIKE ?")
                params.append(f"%{t}%")

        sql = "SELECT prompt, response, tags, ts, embedding FROM interactions"
        if where:
            sql += " WHERE " + " AND ".join(where)
        cur = self.conn.execute(sql, params)
        rows = cur.fetchall()

        if use_embeddings and self.vector_service is not None:
            try:
                q_emb = self.vector_service.vectorise("text", {"text": query})
                scored: list[tuple[float, MemoryEntry]] = []
                for prompt, response, tag_str, ts, emb_json in rows:
                    if not emb_json:
                        continue
                    try:
                        emb = json.loads(emb_json)
                    except Exception:
                        continue
                    tags_raw = [t for t in tag_str.split(",") if t]
                    governed = govern_retrieval(
                        f"{prompt}\n{response}", {"tags": tags_raw}
                    )
                    if governed is None:
                        continue
                    meta, _ = governed
                    tags = meta.pop("tags", tags_raw)
                    score = _cosine_similarity(q_emb, emb)
                    entry = MemoryEntry(
                        redact_secrets(prompt),
                        redact_secrets(response),
                        tags,
                        ts,
                        score,
                        meta,
                    )
                    scored.append((score, entry))
                scored.sort(key=lambda x: x[0], reverse=True)
                return [e for _, e in scored[:limit]]
            except Exception:  # pragma: no cover - embedding is optional
                pass

        results: list[MemoryEntry] = []
        for prompt, response, tag_str, ts, _ in rows:
            if redacted_query.lower() in prompt.lower() or redacted_query.lower() in response.lower():
                tags_raw = [t for t in tag_str.split(",") if t]
                governed = govern_retrieval(
                    f"{prompt}\n{response}", {"tags": tags_raw}
                )
                if governed is None:
                    continue
                meta, _ = governed
                tags = meta.pop("tags", tags_raw)
                results.append(
                    MemoryEntry(
                        redact_secrets(prompt),
                        redact_secrets(response),
                        tags,
                        ts,
                        0.0,
                        meta,
                    )
                )
        return results[:limit]

    def get_similar_entries(
        self,
        query: str,
        *,
        limit: int = 5,
        tags: Sequence[str] | None = None,
        use_embeddings: bool | None = None,
    ) -> List[tuple[float, MemoryEntry]]:
        """Return scored entries most similar to ``query``.

        When ``use_embeddings`` is true and a vector service is available
        cosine similarity between embeddings is used.  Otherwise a simple keyword
        search with a crude relevance score is performed.
        """

        use_embeddings = (
            use_embeddings if use_embeddings is not None else self.vector_service is not None
        )
        entries = self.search_context(
            query,
            limit=limit * 5 if tags and not use_embeddings else limit,
            tags=tags,
            use_embeddings=use_embeddings,
        )

        results: list[tuple[float, MemoryEntry]] = []
        if use_embeddings and self.vector_service is not None:
            for e in entries:
                results.append((e.score, e))
            results.sort(key=lambda x: x[0], reverse=True)
            return results[:limit]

        q = query.lower()
        for e in entries:
            text = f"{e.prompt} {e.response}".lower()
            count = text.count(q)
            score = (count * len(q)) / max(len(text), 1)
            results.append((score, e))
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:limit]

    # ------------------------------------------------------- unified interface
    def store(
        self, key: str, data: str, tags: Sequence[str] | None = None
    ) -> int | None:
        """Persist ``data`` under ``key``.

        The SQLite backend does not expose versions so ``None`` is returned."""

        self.log_interaction(key, data, tags)
        return None

    def retrieve(
        self, query: str, limit: int = 5, tags: Sequence[str] | None = None
    ) -> List[MemoryEntry]:
        """Return stored interactions matching ``query``."""

        return self.search_context(query, limit=limit, tags=tags)

    # -------------------------------------------------------------- compaction
    def compact(self, retention: Mapping[str, int] | int) -> int:
        """Summarise and prune old entries based on a retention policy.

        Parameters
        ----------
        retention:
            Either an ``int`` applied uniformly to all tags or a mapping of
            ``tag -> number of raw entries`` to keep.  Older entries are
            summarised using :func:`_summarise_text` and replaced by a single
            summary entry.  Returns the number of rows removed.
        """

        if isinstance(retention, int):
            cur = self.conn.execute("SELECT tags FROM interactions WHERE tags != ''")
            tags = set()
            for (tag_str,) in cur.fetchall():
                tags.update(t for t in tag_str.split(',') if t)
            retention_map: Dict[str, int] = {t: retention for t in tags}
        else:
            retention_map = dict(retention)

        removed = 0
        for tag, keep in retention_map.items():
            cur = self.conn.execute(
                "SELECT id, prompt, response FROM interactions WHERE tags LIKE ? ORDER BY ts",
                (f"%{tag}%",),
            )
            rows = cur.fetchall()
            if len(rows) <= keep:
                continue

            old_rows = rows[:-keep]
            text = "\n".join(f"{p} {r}" for _, p, r in old_rows)
            summary = _summarise_text(text)
            ts = datetime.utcnow().isoformat()
            self.conn.execute(
                "INSERT INTO interactions(prompt, response, tags, ts, embedding) VALUES (?, ?, ?, ?, NULL)",
                (f"summary:{tag}", summary, f"{tag},summary", ts),
            )
            ids = [str(r[0]) for r in old_rows]
            placeholders = ",".join("?" for _ in ids)
            self.conn.execute(
                f"DELETE FROM interactions WHERE id IN ({placeholders})",
                ids,
            )
            removed += len(ids)

        self.conn.commit()
        return removed

    def prune_old_entries(self, max_rows: int) -> int:
        """Ensure at most ``max_rows`` entries exist for each tag.

        Older entries beyond ``max_rows`` are summarised into a single entry
        and removed.  Returns the number of rows deleted.
        """

        if max_rows <= 0:
            return 0

        cur = self.conn.execute("SELECT tags FROM interactions WHERE tags != ''")
        tags = set()
        for (tag_str,) in cur.fetchall():
            tags.update(t for t in tag_str.split(",") if t)

        removed = 0
        for tag in tags:
            cur = self.conn.execute(
                "SELECT id, prompt, response FROM interactions WHERE tags LIKE ? ORDER BY ts",
                (f"%{tag}%",),
            )
            rows = cur.fetchall()
            if len(rows) <= max_rows:
                continue

            old_rows = rows[:-max_rows]
            text = "\n".join(f"{p} {r}" for _, p, r in old_rows)
            summary = _summarise_text(text)
            ts = datetime.utcnow().isoformat()
            self.conn.execute(
                "INSERT INTO interactions(prompt, response, tags, ts, embedding) VALUES (?, ?, ?, ?, NULL)",
                (f"summary:{tag}", summary, f"{tag},summary", ts),
            )
            ids = [str(r[0]) for r in old_rows]
            placeholders = ",".join("?" for _ in ids)
            self.conn.execute(
                f"DELETE FROM interactions WHERE id IN ({placeholders})",
                ids,
            )
            removed += len(ids)

        self.conn.commit()
        return removed

    # ----------------------------------------------------------------- cleanup
    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:  # pragma: no cover - defensive
            pass

        if self.router is _db_router.GLOBAL_ROUTER:
            try:
                self.router.close()
            except Exception:  # pragma: no cover - defensive
                pass
            _db_router.GLOBAL_ROUTER = None


# ---------------------------------------------------------------------------
# Backwards compatibility wrapper using ``MenaceMemoryManager``


@dataclass
class GPTMemoryRecord:
    prompt: str
    response: str
    tags: List[str]
    ts: str


class GPTMemory(GPTMemoryInterface):
    """Tiny wrapper around :class:`MenaceMemoryManager` used in tests.

    .. deprecated:: 0.1
       Use :class:`GPTMemoryManager` which implements :class:`GPTMemoryInterface`.

    It provides a very small API for storing prompts/responses with
    lightweight tagging.  Only a predefined set of tags is persisted so
    tests can exercise tag filtering behaviour.
    """

    # Legacy wrapper uses the shared tag taxonomy for backwards compatibility
    ALLOWED_TAGS = STANDARD_TAGS

    def __init__(self, manager: MenaceMemoryManager | None = None) -> None:
        if MenaceMemoryManager is None and manager is None:
            raise RuntimeError("MenaceMemoryManager is required")
        warnings.warn(
            "GPTMemory is deprecated; use GPTMemoryManager instead",
            DeprecationWarning,
            stacklevel=2,
        )
        self.manager = manager or MenaceMemoryManager()

    # ------------------------------------------------------------------ logging
    def log_interaction(
        self, prompt: str, response: str, tags: list[str] | None = None
    ) -> int:
        """Store both sides of a conversation.

        Each interaction is logged once for every tag using a key of the form
        ``"gpt:<tag>"`` so that :func:`MenaceMemoryManager.summarise_memory`
        can later condense the history for a given tag.  The underlying memory
        manager assigns version numbers and computes embeddings automatically.

        Parameters
        ----------
        prompt, response:
            Text from the user and the model.
        tags:
            Optional list of labels associated with the interaction.

        Returns
        -------
        int
            The version number from the last stored entry.
        """

        tags = list(tags or [])
        tag_str = ",".join(tags)
        payload = json.dumps({"prompt": prompt, "response": response})

        versions: list[int] = []
        key_tags = tags or ["general"]
        for tag in key_tags:
            key = f"gpt:{tag}"
            versions.append(self.manager.store(key, payload, tags=tag_str))
        return versions[-1] if versions else 0

    # ------------------------------------------------------------------ legacy
    def store(
        self, prompt: str, response: str, tags: Sequence[str] | None = None
    ) -> int:
        """Persist a prompt/response pair.

        This method is retained for backwards compatibility with older tests
        and simply forwards to :meth:`log_interaction` while filtering tags to
        a small predefined allow list.
        """

        valid_tags = [t for t in (tags or []) if t in self.ALLOWED_TAGS]
        return self.log_interaction(prompt, response, list(valid_tags))

    # ------------------------------------------------------------------ context
    def fetch_context(
        self,
        tags: list[str],
        limit: int = 5,
        *,
        scope: Literal["local", "global", "all"] = "local",
    ) -> str:
        """Return a summary of prior interactions for the given ``tags``.

        Summaries are generated by :meth:`MenaceMemoryManager.summarise_memory`
        and are not stored back into the database.
        """

        summaries: list[str] = []
        key_tags = tags or ["general"]
        for tag in key_tags:
            summary = self.manager.summarise_memory(
                f"gpt:{tag}", limit=limit, store=False
            )
            if summary:
                summaries.append(summary)
        return "\n".join(summaries)

    def summarize_and_prune(self, tag: str, limit: int = 20) -> str:
        """Summarise and prune stored interactions for ``tag``.

        This helper delegates to :meth:`MenaceMemoryManager.summarise_memory`
        with ``condense=True`` so that older entries are removed once they have
        been summarised.  The resulting summary is returned.

        Parameters
        ----------
        tag:
            Label identifying the conversation history to prune.
        limit:
            Maximum number of recent entries to include in the summary.
        """

        key = f"gpt:{tag}"
        return self.manager.summarise_memory(key, limit=limit, condense=True)

    def retrieve(
        self, query: str, limit: int = 5, tags: Sequence[str] | None = None
    ) -> List[GPTMemoryRecord]:
        """Return stored interactions matching ``query``.

        Parameters
        ----------
        query:
            Text to search for in stored prompts or responses.
        limit:
            Maximum number of entries to return.
        tags:
            Optional tag filter.  When provided only entries containing one
            of the specified tags are returned.
        """
        entries = self.manager.search(query, limit * 5 if tags else limit)
        wanted = set(tags or [])
        results: List[GPTMemoryRecord] = []
        for e in entries:
            entry_tags = [t for t in e.tags.split(",") if t]
            if wanted and wanted.isdisjoint(entry_tags):
                continue
            try:
                data = json.loads(e.data)
            except Exception:
                continue
            results.append(
                GPTMemoryRecord(
                    data.get("prompt", ""),
                    data.get("response", ""),
                    entry_tags,
                    e.ts,
                )
            )
            if len(results) >= limit:
                break
        return results

    def search_context(
        self,
        query: str,
        *,
        limit: int = 5,
        tags: Sequence[str] | None = None,
        **_: Any,
    ) -> List[GPTMemoryRecord]:
        """Alias for :meth:`retrieve` to satisfy :class:`GPTMemoryInterface`."""

        return self.retrieve(query, limit=limit, tags=tags)


def main(argv: Sequence[str] | None = None) -> None:
    """Simple CLI hook to trigger compaction/pruning tasks."""

    parser = argparse.ArgumentParser(description="Maintain GPT memory store")
    parser.add_argument("--db", default="gpt_memory.db", help="Path to the memory DB")
    parser.add_argument(
        "--keep",
        action="append",
        default=[],
        metavar="TAG=N",
        help="Retention rule; may be supplied multiple times",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    retention: Dict[str, int] = {}
    for item in args.keep:
        tag, _, num = item.partition("=")
        try:
            retention[tag] = int(num)
        except ValueError:
            continue

    mgr = GPTMemoryManager(args.db)
    mgr.compact(retention)
    mgr.close()


__all__ = [
    "GPTMemoryManager",
    "GPTMemory",
    "MemoryEntry",
    "GPTMemoryRecord",
]


# Ensure legacy flat imports resolve to this module instance.
sys.modules["gpt_memory"] = sys.modules[__name__]


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

