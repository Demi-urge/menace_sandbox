from __future__ import annotations

"""Light-weight context aggregation for Menace tasks.

This module exposes :func:`build_context` which queries several internal
knowledge sources and returns a ranked list of relevant context items.
The sources consulted are:

* :mod:`gpt_memory` – semantic search over prior interactions.
* Action and workflow vector stores persisted via ``vector_utils``.
* The :mod:`knowledge_graph` – high level insights and relationships.

Each retrieved text snippet is passed through
:func:`governed_retrieval.govern_retrieval` to ensure licence checks and
secret redaction are applied uniformly.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from governed_retrieval import govern_retrieval

try:  # optional dependencies – best effort imports
    from gpt_memory import GPTMemoryManager
except Exception:  # pragma: no cover - optional
    GPTMemoryManager = None  # type: ignore

try:  # optional dependency used for top insight lookup
    from knowledge_graph import KnowledgeGraph
except Exception:  # pragma: no cover - optional
    KnowledgeGraph = None  # type: ignore

try:  # optional dependency for embeddings
    from vector_service import SharedVectorService
    from vector_utils import cosine_similarity
except Exception:  # pragma: no cover - optional
    SharedVectorService = None  # type: ignore
    cosine_similarity = None  # type: ignore

EMBEDDINGS_PATH = Path("embeddings.jsonl")


@dataclass
class ContextItem:
    """Container for ranked context entries."""

    source: str
    content: str
    score: float
    metadata: Dict[str, Any]


def _load_embeddings(types: Iterable[str]) -> List[Dict[str, Any]]:
    """Return stored embeddings of the requested ``types``.

    The helper reads from ``embeddings.jsonl`` written by
    :func:`vector_utils.persist_embedding`.  Each line is expected to contain
    a JSON object with ``"type"``, ``"id"`` and ``"vector"`` fields.
    """

    results: List[Dict[str, Any]] = []
    if not EMBEDDINGS_PATH.exists():
        return results
    wanted = {t.lower() for t in types}
    with EMBEDDINGS_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("type", "").lower() in wanted:
                results.append(rec)
    return results


def _vector_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity tolerant of mismatched dimensions."""

    n = min(len(a), len(b))
    if not n:
        return 0.0
    dot = sum(a[i] * b[i] for i in range(n))
    norm_a = sum(a[i] * a[i] for i in range(n)) ** 0.5
    norm_b = sum(b[i] * b[i] for i in range(n)) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def _query_vector_sources(description: str, top_n: int) -> List[ContextItem]:
    """Search action and workflow vectors for ``description``."""

    if SharedVectorService is None or cosine_similarity is None:
        return []
    try:
        svc = SharedVectorService()
        q_vec = svc.vectorise("text", {"text": description})
    except Exception:
        return []

    records = _load_embeddings(["action", "workflow"])
    items: List[ContextItem] = []
    for rec in records:
        vec = rec.get("vector") or []
        try:
            similarity = _vector_similarity(q_vec, vec)
        except Exception:
            continue
        text = f"{rec.get('type')}: {rec.get('id')}"
        governed = govern_retrieval(text, {"source": rec.get("type"), "id": rec.get("id")})
        if governed is None:
            continue
        meta, _ = governed
        items.append(
            ContextItem(
                source=str(rec.get("type")),
                content=text,
                score=float(similarity),
                metadata=meta,
            )
        )
    items.sort(key=lambda x: x.score, reverse=True)
    return items[:top_n]


def _query_memory(description: str, top_n: int) -> List[ContextItem]:
    if GPTMemoryManager is None:
        return []
    try:
        mgr = GPTMemoryManager()
    except Exception:
        return []
    results: List[ContextItem] = []
    try:
        entries = mgr.get_similar_entries(description, limit=top_n, use_embeddings=True)
    except Exception:
        entries = []
    for score, entry in entries:
        text = f"{entry.prompt}\n{entry.response}"
        governed = govern_retrieval(text, {"tags": entry.tags})
        if governed is None:
            continue
        meta, _ = governed
        meta.update(entry.metadata or {})
        results.append(
            ContextItem(
                source="gpt_memory",
                content=text,
                score=float(score),
                metadata=meta,
            )
        )
    return results


def _query_knowledge_graph(description: str, top_n: int) -> List[ContextItem]:
    if KnowledgeGraph is None:
        return []
    try:
        graph = KnowledgeGraph()
    except Exception:
        return []
    tokens = set(description.lower().split())
    results: List[ContextItem] = []
    try:
        insights = graph.top_insights(limit=top_n * 2)
    except Exception:
        insights = []
    for key, neighbors in insights:
        text = f"{key} {' '.join(neighbors)}".strip()
        score = len(tokens.intersection(key.lower().split()))
        governed = govern_retrieval(text, {"source": "knowledge_graph", "neighbors": neighbors})
        if governed is None:
            continue
        meta, _ = governed
        results.append(
            ContextItem(
                source="knowledge_graph",
                content=text,
                score=float(score),
                metadata=meta,
            )
        )
    results.sort(key=lambda x: x.score, reverse=True)
    return results[:top_n]


def build_context(task_description: str, *, top_n: int = 5) -> List[ContextItem]:
    """Return ``top_n`` ranked context items for ``task_description``."""

    items: List[ContextItem] = []
    items.extend(_query_memory(task_description, top_n))
    items.extend(_query_vector_sources(task_description, top_n))
    items.extend(_query_knowledge_graph(task_description, top_n))
    items.sort(key=lambda x: x.score, reverse=True)
    return items[:top_n]


__all__ = ["build_context", "ContextItem"]
