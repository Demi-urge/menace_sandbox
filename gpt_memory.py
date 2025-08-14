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

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import sqlite3
import argparse
from typing import Any, List, Sequence, Mapping, Dict

try:  # Optional dependency used for semantic embeddings
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - keep import lightweight
    SentenceTransformer = None  # type: ignore

try:  # Optional dependency used by the light wrapper ``GPTMemory``
    from menace_memory_manager import MenaceMemoryManager, _summarise_text  # type: ignore
except Exception:  # pragma: no cover - tests stub this module
    MenaceMemoryManager = None  # type: ignore

    def _summarise_text(text: str, ratio: float = 0.2) -> str:  # pragma: no cover - fallback
        """Fallback summariser used when menace_memory_manager is unavailable."""
        return text[: max(1, int(len(text) * ratio))]


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


class GPTMemoryManager:
    """Persist and query GPT interactions using SQLite.

    Parameters
    ----------
    db_path:
        Location of the SQLite database.  ``"gpt_memory.db"`` by default.
    embedder:
        Optional :class:`SentenceTransformer` instance.  When provided each
        prompt is embedded and semantic search can be performed.
    """

    def __init__(
        self,
        db_path: str | Path = "gpt_memory.db",
        *,
        embedder: SentenceTransformer | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.embedder = embedder
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
                embedding TEXT
            )
            """
        )
        self.conn.commit()

    # --------------------------------------------------------------- interface
    def log_interaction(
        self,
        prompt: str,
        response: str,
        tags: Sequence[str] | None = None,
    ) -> None:
        """Record a GPT interaction in persistent storage."""

        timestamp = datetime.utcnow().isoformat()
        tag_str = ",".join(tags) if tags else ""
        embedding: str | None = None
        if self.embedder is not None:
            try:
                vec = self.embedder.encode(prompt)
                embedding = json.dumps([float(x) for x in vec])
            except Exception:  # pragma: no cover - embedding is optional
                embedding = None

        self.conn.execute(
            "INSERT INTO interactions(prompt, response, tags, ts, embedding)"
            " VALUES (?, ?, ?, ?, ?)",
            (prompt, response, tag_str, timestamp, embedding),
        )
        self.conn.commit()

    def search_context(
        self,
        query: str,
        *,
        limit: int = 5,
        tags: Sequence[str] | None = None,
        use_embeddings: bool = False,
    ) -> List[MemoryEntry]:
        """Search stored interactions matching ``query``.

        When ``use_embeddings`` is true and an embedder is available cosine
        similarity between the query and stored prompts is used; otherwise a
        simple substring search over prompt/response is performed.
        """

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

        if use_embeddings and self.embedder is not None:
            try:
                q_emb = self.embedder.encode(query)
                scored: list[tuple[float, MemoryEntry]] = []
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
                scored.sort(key=lambda x: x[0], reverse=True)
                return [e for _, e in scored[:limit]]
            except Exception:  # pragma: no cover - embedding is optional
                pass

        results: list[MemoryEntry] = []
        for prompt, response, tag_str, ts, _ in rows:
            if query.lower() in prompt.lower() or query.lower() in response.lower():
                results.append(
                    MemoryEntry(prompt, response, tag_str.split(",") if tag_str else [], ts)
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

        When ``use_embeddings`` is true and an embedder is available cosine
        similarity between embeddings is used.  Otherwise a simple keyword
        search with a crude relevance score is performed.
        """

        use_embeddings = (
            use_embeddings if use_embeddings is not None else self.embedder is not None
        )
        entries = self.search_context(
            query,
            limit=limit * 5 if tags and not use_embeddings else limit,
            tags=tags,
            use_embeddings=use_embeddings,
        )

        results: list[tuple[float, MemoryEntry]] = []
        if use_embeddings and self.embedder is not None:
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

    # ----------------------------------------------------------------- cleanup
    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:  # pragma: no cover - defensive
            pass


# ---------------------------------------------------------------------------
# Backwards compatibility wrapper using ``MenaceMemoryManager``


@dataclass
class GPTMemoryRecord:
    prompt: str
    response: str
    tags: List[str]
    ts: str


class GPTMemory:
    """Tiny wrapper around :class:`MenaceMemoryManager` used in tests."""

    ALLOWED_TAGS = {"improvement", "bugfix", "insight"}

    def __init__(self, manager: MenaceMemoryManager | None = None) -> None:
        if MenaceMemoryManager is None and manager is None:
            raise RuntimeError("MenaceMemoryManager is required")
        self.manager = manager or MenaceMemoryManager()

    def store(
        self, prompt: str, response: str, tags: Sequence[str] | None = None
    ) -> int:
        valid_tags = [t for t in (tags or []) if t in self.ALLOWED_TAGS]
        key = f"gpt:{datetime.utcnow().isoformat()}"
        data = json.dumps({"prompt": prompt, "response": response})
        tag_str = ",".join(valid_tags)
        return self.manager.store(key, data, tags=tag_str)

    def retrieve(
        self, query: str, limit: int = 5, tags: Sequence[str] | None = None
    ) -> List[GPTMemoryRecord]:
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


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

