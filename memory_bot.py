"""Memory Bot for storing and retrieving conversation data."""

from __future__ import annotations

from .bot_registry import BotRegistry
from .data_bot import DataBot

from .coding_bot_interface import self_coding_managed
import gzip
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from gpt_memory_interface import GPTMemoryInterface

registry = BotRegistry()
data_bot = DataBot(start_server=False)

try:
    from security.secret_redactor import redact_secrets
except Exception:  # pragma: no cover - fallback for legacy path
    from secret_redactor import redact_secrets  # type: ignore

from governed_embeddings import governed_embed

try:  # optional dependency for embeddings
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional
    SentenceTransformer = None  # type: ignore

try:
    from pymongo import MongoClient  # type: ignore
except Exception:  # pragma: no cover - optional
    MongoClient = None  # type: ignore


@dataclass
class MemoryRecord:
    user: str
    text: str
    ts: str = datetime.utcnow().isoformat()
    meta: Dict[str, Any] | None = None


class MemoryStorage:
    """Simple storage backend using MongoDB or gzipped JSON."""

    def __init__(self, path: Path | str = "memory.json.gz", mongo_url: str | None = None) -> None:
        self.path = Path(path)
        self.mongo_url = mongo_url
        self._client = MongoClient(mongo_url) if MongoClient and mongo_url else None
        if self._client:
            self.col = self._client.get_database()["mem"]
        else:
            self.col = None
            if not self.path.exists():
                with gzip.open(self.path, "wt", encoding="utf-8") as fh:
                    json.dump([], fh)

    def add(self, rec: MemoryRecord) -> None:
        if self.col:
            self.col.insert_one(rec.__dict__)
            return
        with gzip.open(self.path, "rt", encoding="utf-8") as fh:
            data = json.load(fh)
        data.append(rec.__dict__)
        with gzip.open(self.path, "wt", encoding="utf-8") as fh:
            json.dump(data, fh)

    def query(self, text: str, limit: int = 10) -> List[MemoryRecord]:
        if self.col:
            cur = self.col.find({"text": {"$regex": text, "$options": "i"}}).sort("ts", -1).limit(limit)
            return [MemoryRecord(**d) for d in cur]
        with gzip.open(self.path, "rt", encoding="utf-8") as fh:
            data = json.load(fh)
        res = [d for d in data if text.lower() in d["text"].lower()]
        res = sorted(res, key=lambda d: d["ts"], reverse=True)[:limit]
        return [MemoryRecord(**d) for d in res]


class VectorMemoryStorage(MemoryStorage):
    """Memory storage that auto-embeds text for vector search."""

    def __init__(
        self,
        path: Path | str = "memory.json.gz",
        mongo_url: str | None = None,
        embedder: SentenceTransformer | None = None,
    ) -> None:
        super().__init__(path, mongo_url)
        self.embedder = embedder or (
            SentenceTransformer("all-MiniLM-L6-v2") if SentenceTransformer else None
        )

    def _embed(self, text: str) -> Optional[List[float]]:
        if not self.embedder:
            return None
        return governed_embed(text, self.embedder)

    def add(self, rec: MemoryRecord) -> None:  # type: ignore[override]
        original = rec.text.strip()
        if not original:
            return
        rec.text = redact_secrets(original)
        embedding = self._embed(original)
        if embedding is not None:
            meta = rec.meta or {}
            meta["embedding"] = embedding
            rec.meta = meta
        super().add(rec)

    def query_vector(self, text: str, limit: int = 5) -> List[MemoryRecord]:
        embedding = self._embed(text)
        redacted_query = redact_secrets(text)
        if embedding is None:
            results = self.query(redacted_query, limit)
        else:
            if self.col:
                cur = self.col.find({"meta.embedding": {"$exists": True}})
                data = list(cur)
            else:
                with gzip.open(self.path, "rt", encoding="utf-8") as fh:
                    data = json.load(fh)
            scored: List[tuple[float, Dict[str, Any]]] = []
            for row in data:
                emb = row.get("meta", {}).get("embedding")
                if not emb:
                    continue
                score = sum(a * b for a, b in zip(embedding, emb))
                scored.append((score, row))
            scored.sort(key=lambda x: x[0], reverse=True)
            results = [MemoryRecord(**row) for _, row in scored[:limit]]
        for r in results:
            r.text = redact_secrets(r.text)
        return results


@self_coding_managed(bot_registry=registry, data_bot=data_bot)
class MemoryBot(GPTMemoryInterface):
    """Bot that stores conversations and provides search with caching."""

    def __init__(self, storage: MemoryStorage | None = None) -> None:
        self.storage = storage or MemoryStorage()
        self.cache: Dict[str, List[MemoryRecord]] = {}

    def log(self, user: str, text: str, meta: Optional[Dict[str, Any]] = None) -> None:
        rec = MemoryRecord(user=user, text=text, meta=meta)
        self.storage.add(rec)

    def search(self, text: str, limit: int = 5) -> List[MemoryRecord]:
        key = f"{text}:{limit}"
        if key in self.cache:
            return self.cache[key]
        res = self.storage.query(text, limit)
        self.cache[key] = res
        return res

    def prime(self) -> None:
        """Prepare the bot for heavy usage by clearing cache."""
        self.cache.clear()

    # ------------------------------------------------------- unified interface
    def log_interaction(
        self, prompt: str, response: str, tags: Sequence[str] | None = None
    ) -> None:
        meta = {"tags": list(tags or [])} if tags else None
        self.log("user", prompt, meta)
        self.log("assistant", response, meta)

    def store(
        self, key: str, data: str, tags: Sequence[str] | None = None
    ) -> int | None:
        self.log_interaction(key, data, tags)
        return None

    def retrieve(
        self, query: str, limit: int = 5, tags: Sequence[str] | None = None
    ) -> List[MemoryRecord]:
        results = self.search(query, limit)
        if tags:
            wanted = set(tags)
            results = [
                r
                for r in results
                if not wanted.isdisjoint(r.meta.get("tags", []) if r.meta else [])
            ]
        return results

    def search_context(
        self,
        query: str,
        *,
        limit: int = 5,
        tags: Sequence[str] | None = None,
        **_: Any,
    ) -> List[MemoryRecord]:
        return self.retrieve(query, limit=limit, tags=tags)


__all__ = ["MemoryRecord", "MemoryStorage", "VectorMemoryStorage", "MemoryBot"]