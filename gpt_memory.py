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
from typing import Any, List, Optional, Sequence

try:
    from menace_memory_manager import MenaceMemoryManager, _summarise_text  # type: ignore
except Exception:  # pragma: no cover - fallback summariser
    MenaceMemoryManager = None  # type: ignore
    def _summarise_text(text: str, ratio: float = 0.2) -> str:
        text = text.strip()
        if not text:
            return ""
        sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
        if len(sentences) <= 1:
            return text
        count = max(1, int(len(sentences) * ratio))
        return ". ".join(sentences[:count]) + "."

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
        compact_every: int | None = 100,
        retention_policy: dict[str, int] | None = None,
        default_retention: int = 50,
    ) -> None:
        self.path = Path(path)
        self.conn = sqlite3.connect(self.path)
        self.embedder = embedder
        self._ensure_schema()
        self.compact_every = max(1, compact_every) if compact_every else 0
        self._log_counter = 0
        self.retention_policy = retention_policy or {}
        self.default_retention = max(0, default_retention)

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
        self._log_counter += 1
        if self.compact_every and self._log_counter % self.compact_every == 0:
            self.compact()

    def compact(
        self,
        *,
        retention_policy: dict[str, int] | None = None,
        default_keep: int | None = None,
    ) -> None:
        """Summarise old entries per tag and prune them."""
        policy = retention_policy or self.retention_policy
        keep_default = self.default_retention if default_keep is None else max(0, default_keep)
        cur = self.conn.execute("SELECT DISTINCT tags FROM interactions")
        tags = set()
        for (tag_str,) in cur.fetchall():
            if not tag_str:
                tags.add("")
            else:
                for t in tag_str.split(","):
                    tags.add(t.strip())
        for tag in tags:
            keep = policy.get(tag, keep_default)
            if tag:
                q = "SELECT id, prompt, response FROM interactions WHERE ',' || tags || ',' LIKE ? ORDER BY ts DESC"
                rows = self.conn.execute(q, (f'%,{tag},%',)).fetchall()
            else:
                q = "SELECT id, prompt, response FROM interactions WHERE tags = '' OR tags IS NULL ORDER BY ts DESC"
                rows = self.conn.execute(q).fetchall()
            if len(rows) <= keep:
                continue
            to_summarise = rows[keep:]
            text = "\n".join(f"Q: {p}\nA: {r}" for _, p, r in to_summarise)
            summary = _summarise_text(text)
            ids = [rid for rid, _, _ in to_summarise]
            placeholders = ",".join("?" for _ in ids)
            self.conn.execute(f"DELETE FROM interactions WHERE id IN ({placeholders})", ids)
            tag_str = ",".join(filter(None, [tag, "summary"]))
            self.conn.execute(
                "INSERT INTO interactions(prompt, response, tags, ts, embedding) VALUES (?, ?, ?, ?, NULL)",
                (f"Summary of {tag}" if tag else "Summary", summary, tag_str, datetime.utcnow().isoformat()),
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

@dataclass
class GPTMemoryRecord:
    prompt: str
    response: str
    tags: List[str]
    ts: str


class GPTMemory:
    """Wrapper around MenaceMemoryManager for simple store/retrieve."""

    ALLOWED_TAGS = {"improvement", "bugfix", "insight"}

    def __init__(self, manager: MenaceMemoryManager | None = None) -> None:
        if MenaceMemoryManager is None and manager is None:
            raise RuntimeError("MenaceMemoryManager is required")
        self.manager = manager or MenaceMemoryManager()

    def store(self, prompt: str, response: str, tags: Sequence[str] | None = None) -> int:
        tag_list = [t for t in (tags or []) if t in self.ALLOWED_TAGS]
        key = f"gpt:{datetime.utcnow().isoformat()}"
        data = json.dumps({"prompt": prompt, "response": response})
        tag_str = ",".join(tag_list)
        return self.manager.store(key, data, tags=tag_str)

    def retrieve(self, query: str, limit: int = 5, tags: Sequence[str] | None = None) -> List[GPTMemoryRecord]:
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
            results.append(GPTMemoryRecord(data.get("prompt", ""), data.get("response", ""), entry_tags, e.ts))
            if len(results) >= limit:
                break
        return results

def cli(argv: Sequence[str] | None = None) -> int:
    """Simple command line interface for maintenance tasks."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_prune = sub.add_parser("prune", help="Compact/prune old memory entries")
    p_prune.add_argument("--db", default="gpt_memory.db", help="Database file")
    p_prune.add_argument(
        "--keep", type=int, default=50, help="Default number of raw entries to keep per tag"
    )
    p_prune.add_argument(
        "--policy",
        action="append",
        default=[],
        metavar="TAG=N",
        help="Per-tag retention policy",
    )

    args = parser.parse_args(argv)

    if args.cmd == "prune":
        policy: dict[str, int] = {}
        for item in args.policy:
            if "=" in item:
                tag, val = item.split("=", 1)
                try:
                    policy[tag] = int(val)
                except ValueError:
                    continue
        mgr = GPTMemoryManager(args.db)
        mgr.compact(retention_policy=policy or None, default_keep=args.keep)
        mgr.close()
        return 0
    return 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(cli())


__all__ = ["GPTMemoryManager", "MemoryEntry", "GPTMemory", "GPTMemoryRecord", "cli"]
