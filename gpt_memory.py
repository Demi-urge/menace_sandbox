"""Lightweight GPT interaction memory manager.

This module provides :class:`GPTMemoryManager` which records GPT prompt/response
pairs in a tiny SQLite database.  The manager can optionally store sentence
embeddings using `sentence_transformers` allowing semantic search over previous
interactions.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import sqlite3
from typing import List, Optional, Sequence, Any

try:  # optional dependency
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - embeddings are optional
    SentenceTransformer = None  # type: ignore


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the cosine similarity between two vectors."""
    import math
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    denom = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b))
    return dot / denom if denom else 0.0


@dataclass
class MemoryEntry:
    """Simple representation of a stored interaction."""
    prompt: str
    response: str
    tags: List[str]
    timestamp: str
    score: float = 0.0


class GPTMemoryManager:
    """Persist and query GPT interactions.

    Parameters
    ----------
    path:
        Location of the SQLite database file.  ``"gpt_memory.db"`` by default.
    embedder:
        Optional ``SentenceTransformer`` instance used to generate vector
        embeddings.  When provided, :meth:`search_context` can perform semantic
        lookups using cosine similarity.
    """

    def __init__(
        self,
        path: str | Path = "gpt_memory.db",
        *,
        embedder: SentenceTransformer | None = None,
    ) -> None:
        self.path = Path(path)
        self.conn = sqlite3.connect(self.path)
        self.embedder = embedder
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                tags TEXT,
                ts TEXT NOT NULL,
                embedding TEXT
            )
            """
        )
        self.conn.commit()

    def log_interaction(
        self,
        prompt: str,
        response: str,
        tags: Optional[Sequence[str]] = None,
    ) -> None:
        """Record a GPT interaction."""
        ts = datetime.utcnow().isoformat()
        tag_str = ",".join(tags) if tags else ""
        embedding: str | None = None
        if self.embedder is not None:
            try:
                emb = self.embedder.encode(prompt)
                embedding = json.dumps([float(x) for x in emb])
            except Exception:
                embedding = None
        self.conn.execute(
            "INSERT INTO interactions(prompt, response, tags, ts, embedding) VALUES (?, ?, ?, ?, ?)",
            (prompt, response, tag_str, ts, embedding),
        )
        self.conn.commit()

    def get_similar_entries(
        self,
        query: str,
        *,
        limit: int = 5,
        method: str = "auto",
        tags: Optional[Sequence[str]] = None,
    ) -> List[tuple[float, MemoryEntry]]:
        """Retrieve top-N similar memory entries.

        Parameters
        ----------
        query:
            Text used for similarity search.
        limit:
            Maximum number of entries to return.
        method:
            ``"vector"`` to force embedding search, ``"text"`` for substring
            matching, or ``"auto"`` (default) which prefers embeddings when
            available.
        tags:
            Optional list of tags to filter the search.

        Returns
        -------
        List of ``(score, MemoryEntry)`` tuples sorted by descending score.
        """

        where: List[str] = []
        params: List[Any] = []
        if tags:
            for t in tags:
                where.append("tags LIKE ?")
                params.append(f"%{t}%")
        base_query = "SELECT prompt, response, tags, ts, embedding FROM interactions"
        if where:
            base_query += " WHERE " + " AND ".join(where)
        cur = self.conn.execute(base_query, params)
        rows = cur.fetchall()

        scored: List[tuple[float, MemoryEntry]] = []
        use_vector = method in ("vector", "auto") and self.embedder is not None
        q_emb: Sequence[float] | None = None
        if use_vector:
            try:
                q_emb = self.embedder.encode(query)
            except Exception:
                use_vector = False

        if use_vector and q_emb is not None:
            for prompt, response, tag_str, ts, emb_json in rows:
                if not emb_json:
                    continue
                try:
                    emb = json.loads(emb_json)
                except Exception:
                    continue
                score = _cosine_similarity(q_emb, emb)
                entry = MemoryEntry(
                    prompt, response, tag_str.split(",") if tag_str else [], ts, score
                )
                scored.append((score, entry))
        else:
            import difflib

            for prompt, response, tag_str, ts, _ in rows:
                text = f"{prompt} {response}"
                score = difflib.SequenceMatcher(
                    None, query.lower(), text.lower()
                ).ratio()
                entry = MemoryEntry(
                    prompt, response, tag_str.split(",") if tag_str else [], ts, score
                )
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:limit]

    def search_context(
        self,
        query: str,
        *,
        tags: Optional[Sequence[str]] = None,
        limit: int = 5,
        use_embeddings: bool = True,
    ) -> List[MemoryEntry]:
        """Search stored interactions.

        The search will use cosine similarity over embeddings when an embedder
        is available and ``use_embeddings`` is ``True``.  Otherwise a simple
        LIKE based text search is performed.
        """
        where: List[str] = []
        params: List[Any] = []
        if tags:
            for t in tags:
                where.append("tags LIKE ?")
                params.append(f"%{t}%")
        base_query = "SELECT prompt, response, tags, ts, embedding FROM interactions"
        if where:
            base_query += " WHERE " + " AND ".join(where)
        cur = self.conn.execute(base_query, params)
        rows = cur.fetchall()
        if use_embeddings and self.embedder is not None:
            try:
                q_emb = self.embedder.encode(query)
                scored = []
                for prompt, response, tag_str, ts, emb_json in rows:
                    if not emb_json:
                        continue
                    try:
                        emb = json.loads(emb_json)
                    except Exception:
                        continue
                    score = _cosine_similarity(q_emb, emb)
                    entry = MemoryEntry(prompt, response, tag_str.split(",") if tag_str else [], ts)
                    scored.append((score, entry))
                scored.sort(key=lambda x: x[0], reverse=True)
                return [e for _, e in scored[:limit]]
            except Exception:
                pass
        results: List[MemoryEntry] = []
        for prompt, response, tag_str, ts, _ in rows:
            if query.lower() in prompt.lower() or query.lower() in response.lower():
                results.append(MemoryEntry(prompt, response, tag_str.split(",") if tag_str else [], ts))
        return results[:limit]

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def __del__(self) -> None:  # pragma: no cover
        self.close()


__all__ = ["GPTMemoryManager", "MemoryEntry"]
