"""Query Bot for natural language queries across Menace components."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import os
import logging

try:
    import spacy  # type: ignore
except Exception:  # pragma: no cover - optional
    spacy = None  # type: ignore

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover - optional
    redis = None  # type: ignore

logger = logging.getLogger(__name__)

from .chatgpt_idea_bot import ChatGPTClient
from . import database_manager


@dataclass
class QueryResult:
    text: str
    data: Dict[str, Any] | None = None


class ContextStore:
    """Simple in-memory or Redis based context storage."""

    def __init__(self, redis_url: str | None = None) -> None:
        self.redis_url = redis_url
        if redis and redis_url:
            try:
                self.client = redis.from_url(redis_url)
            except Exception:  # pragma: no cover - optional
                self.client = None
        else:
            self.client = None
        self.memory: Dict[str, List[str]] = {}

    def add(self, cid: str, text: str) -> None:
        if self.client:
            try:
                self.client.rpush(cid, text)
                return
            except Exception:
                logger.exception("redis rpush failed")
                logger.warning("falling back to memory store")
        self.memory.setdefault(cid, []).append(text)

    def history(self, cid: str) -> List[str]:
        if self.client:
            try:
                values = self.client.lrange(cid, 0, -1)
                return [v.decode() if isinstance(v, bytes) else str(v) for v in values]
            except Exception:
                logger.exception("redis lrange failed")
                logger.warning("falling back to memory store")
        return list(self.memory.get(cid, []))


class SimpleNLU:
    """Lightweight natural language understanding helper."""

    def __init__(self) -> None:
        self.nlp = spacy.load("en_core_web_sm") if spacy else None

    def parse(self, text: str) -> Dict[str, Any]:
        """Parse ``text`` into an intent and entities."""

        if self.nlp:
            doc = self.nlp(text)
            entities = [(ent.label_, ent.text) for ent in doc.ents]
            intent = doc[0].lemma_ if doc else "query"
            return {"intent": intent, "entities": entities}

        tokens = [t.strip() for t in text.lower().split()]
        intent = tokens[0] if tokens else "query"
        entities = []
        for tok in tokens[1:]:
            if tok.isdigit():
                entities.append(("NUMBER", tok))
            elif tok.isalpha():
                entities.append(("WORD", tok))
        return {"intent": intent, "entities": entities}


class DataFetcher:
    """Retrieve records from the models database."""

    def __init__(self, data: Optional[Dict[str, Dict[str, Any]]] = None, db_path: Path | None = None) -> None:
        self.data = data or {}
        self.db_path = db_path or database_manager.DB_PATH

    def fetch(self, entities: Iterable[str]) -> Dict[str, Any]:
        res = {}
        for e in entities:
            if e in self.data:
                res[e] = self.data[e]
                continue
            rows = database_manager.search_models(e, db_path=self.db_path)
            if rows:
                res[e] = rows
        return res


class QueryBot:
    """NLU-driven query handler with optional context management."""

    def __init__(
        self,
        client: ChatGPTClient | None = None,
        fetcher: DataFetcher | None = None,
        store: ContextStore | None = None,
        nlu: SimpleNLU | None = None,
    ) -> None:
        if client is None:
            api_key = os.environ.get("OPENAI_API_KEY", "")
            client = ChatGPTClient(api_key)
        self.client = client
        self.fetcher = fetcher or DataFetcher()
        self.store = store or ContextStore()
        self.nlu = nlu or SimpleNLU()

    def process(self, query: str, context_id: str) -> QueryResult:
        parsed = self.nlu.parse(query)
        ents = [t[1] for t in parsed.get("entities", [])]
        data = self.fetcher.fetch(ents)
        self.store.add(context_id, query)
        prompt = f"Summarize the following data: {json.dumps(data)}"
        answer = self.client.ask([{"role": "user", "content": prompt}])
        text = (
            answer.get("choices", [{}])[0].get("message", {}).get("content", "")
        )
        return QueryResult(text=text, data=data)

    def history(self, context_id: str) -> List[str]:
        return self.store.history(context_id)


__all__ = ["QueryBot", "QueryResult", "ContextStore", "DataFetcher"]
