"""Query Bot for natural language queries across Menace components."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import logging
import uuid

try:  # pragma: no cover - support package and standalone usage
    from config import get_config  # type: ignore
except Exception:  # pragma: no cover - fallback to package import
    from menace.config import get_config  # type: ignore

try:
    import spacy  # type: ignore
except Exception:  # pragma: no cover - optional
    spacy = None  # type: ignore

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover - optional
    redis = None  # type: ignore

from .chatgpt_idea_bot import ChatGPTClient
from vector_service.context_builder import ContextBuilder
from gpt_memory_interface import GPTMemoryInterface
from . import database_manager
from snippet_compressor import compress_snippets
try:  # memory-aware wrapper
    from .memory_aware_gpt_client import ask_with_memory
except Exception:  # pragma: no cover - fallback for flat layout
    from memory_aware_gpt_client import ask_with_memory  # type: ignore
try:  # canonical tag constants
    from .log_tags import FEEDBACK, IMPROVEMENT_PATH, ERROR_FIX, INSIGHT
except Exception:  # pragma: no cover - fallback for flat layout
    from log_tags import FEEDBACK, IMPROVEMENT_PATH, ERROR_FIX, INSIGHT  # type: ignore
try:  # shared GPT memory instance
    from .shared_gpt_memory import GPT_MEMORY_MANAGER
except Exception:  # pragma: no cover - fallback for flat layout
    from shared_gpt_memory import GPT_MEMORY_MANAGER  # type: ignore
try:  # pragma: no cover - allow flat imports
    from .local_knowledge_module import LocalKnowledgeModule
except Exception:  # pragma: no cover - fallback for flat layout
    from local_knowledge_module import LocalKnowledgeModule  # type: ignore
try:
    from .run_autonomous import LOCAL_KNOWLEDGE_MODULE as _LOCAL_KNOWLEDGE
except Exception:
    try:
        from .sandbox_runner import LOCAL_KNOWLEDGE_MODULE as _LOCAL_KNOWLEDGE
    except Exception:  # pragma: no cover - fallback
        _LOCAL_KNOWLEDGE = None
if _LOCAL_KNOWLEDGE is None:
    _LOCAL_KNOWLEDGE = LocalKnowledgeModule(manager=GPT_MEMORY_MANAGER)
LOCAL_KNOWLEDGE_MODULE = _LOCAL_KNOWLEDGE

logger = logging.getLogger(__name__)


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

    def __init__(
        self,
        data: Optional[Dict[str, Dict[str, Any]]] = None,
        db_path: Path | None = None,
    ) -> None:
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
        gpt_memory: GPTMemoryInterface | None = GPT_MEMORY_MANAGER,
        knowledge: LocalKnowledgeModule | None = LOCAL_KNOWLEDGE_MODULE,
        *,
        context_builder: ContextBuilder,
    ) -> None:
        if client is None:
            api_key = get_config().api_keys.openai
            client = ChatGPTClient(
                api_key,
                gpt_memory=gpt_memory or GPT_MEMORY_MANAGER,
                context_builder=context_builder,
            )
        self.client = client
        self.context_builder = context_builder
        try:
            self.client.context_builder = context_builder
        except Exception:
            logger.debug(
                "failed to attach context_builder to client", exc_info=True
            )
        self.context_builder.refresh_db_weights()
        self.fetcher = fetcher or DataFetcher()
        self.store = store or ContextStore()
        self.nlu = nlu or SimpleNLU()
        self.gpt_memory = gpt_memory
        self.local_knowledge = knowledge or LocalKnowledgeModule(manager=self.gpt_memory)
        if getattr(self.client, "gpt_memory", None) is None:
            try:
                self.client.gpt_memory = self.gpt_memory
            except Exception:
                logger.debug("failed to attach gpt_memory to client", exc_info=True)

    def process(self, query: str, context_id: str) -> QueryResult:
        parsed = self.nlu.parse(query)
        ents = [t[1] for t in parsed.get("entities", [])]
        data = self.fetcher.fetch(ents)
        self.store.add(context_id, query)
        try:
            _, meta = self.context_builder.build(
                query, session_id=uuid.uuid4().hex, return_metadata=True
            )
            compressed = {
                k: [compress_snippets(m) for m in v]
                for k, v in meta.items()
            }
            vec_prompt = json.dumps(compressed)
        except Exception:
            vec_prompt = "{}"
        prompt = f"Summarize the following data: {json.dumps(data)}\nContext: {vec_prompt}"
        answer = ask_with_memory(
            self.client,
            "query_bot.process",
            prompt,
            memory=self.local_knowledge,
            context_builder=self.context_builder,
            tags=[FEEDBACK, IMPROVEMENT_PATH, ERROR_FIX, INSIGHT],
        )
        text = (
            answer.get("choices", [{}])[0].get("message", {}).get("content", "")
        )
        return QueryResult(text=text, data=data)

    def history(self, context_id: str) -> List[str]:
        return self.store.history(context_id)


__all__ = ["QueryBot", "QueryResult", "ContextStore", "DataFetcher"]
