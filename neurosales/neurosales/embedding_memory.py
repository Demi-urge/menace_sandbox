from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional
import os

try:
    import faiss  # type: ignore
    from sentence_transformers import SentenceTransformer
    import numpy as np
except Exception:  # pragma: no cover - optional heavy deps
    faiss = None
    SentenceTransformer = None
    np = None


@dataclass
class EmbeddedMessage:
    timestamp: float
    role: str
    content: str
    embedding: List[float]


@dataclass
class EmbeddingConversationMemory:
    """Conversation memory using embeddings and FAISS for fast lookup."""

    max_messages: int = 5
    ttl_seconds: Optional[int] = None
    model_name: str = "all-MiniLM-L6-v2"
    _messages: deque[EmbeddedMessage] = field(default_factory=deque, init=False)
    _model: Optional[SentenceTransformer] = field(default=None, init=False)
    _index: Optional[faiss.Index] = field(default=None, init=False)

    def __post_init__(self) -> None:
        if SentenceTransformer is None or faiss is None:
            return
        self._model = SentenceTransformer(self.model_name)
        dim = self._model.get_sentence_embedding_dimension()
        self._index = faiss.IndexFlatL2(dim)

    def _prune(self) -> None:
        if self.ttl_seconds is None:
            return
        expiry = time.time() - self.ttl_seconds
        while self._messages and self._messages[0].timestamp < expiry:
            self._messages.popleft()
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        if self._index is None or np is None:
            return
        self._index.reset()
        if not self._messages:
            return
        embeddings = np.stack([m.embedding for m in self._messages]).astype("float32")
        self._index.add(embeddings)

    def add_message(self, role: str, content: str) -> None:
        if SentenceTransformer is None or faiss is None or np is None:
            msg = EmbeddedMessage(time.time(), role, content, [])
            self._messages.append(msg)
            if len(self._messages) > self.max_messages:
                self._messages.popleft()
            self._prune()
            return

        embedding = self._model.encode(content, convert_to_numpy=True).astype("float32")
        msg = EmbeddedMessage(time.time(), role, content, embedding)
        self._messages.append(msg)
        if len(self._messages) > self.max_messages:
            self._messages.popleft()
        self._prune()
        self._rebuild_index()

    def get_recent_messages(self) -> List[EmbeddedMessage]:
        return list(self._messages)

    def most_similar(self, text: str, top_k: int = 3) -> List[EmbeddedMessage]:
        if self._index is None or self._model is None or np is None:
            return []
        if not self._messages:
            return []
        query = self._model.encode(text, convert_to_numpy=True).astype("float32")
        D, indices = self._index.search(
            query.reshape(1, -1), min(top_k, len(self._messages))
        )
        return [self._messages[i] for i in indices[0]]


@dataclass
class DatabaseEmbeddingMemory(EmbeddingConversationMemory):
    """Embedding memory persisted to a SQL database."""

    session_factory: Optional[callable] = None
    db_url: Optional[str] = None

    def __post_init__(self) -> None:  # type: ignore[override]
        super().__post_init__()
        if self.session_factory is None:
            from .sql_db import create_session as create_sql_session, ensure_schema

            ensure_schema(self.db_url or os.environ.get("NEURO_DB_URL", "sqlite://"))
            self.session_factory = create_sql_session(self.db_url)

        from .sql_db import EmbeddingMessage

        Session = self.session_factory
        self._messages = deque()
        with Session() as s:
            rows = (
                s.query(EmbeddingMessage)
                .order_by(EmbeddingMessage.timestamp.desc())
                .limit(self.max_messages)
                .all()
            )
        for r in reversed(rows):
            emb = r.embedding or []
            self._messages.append(EmbeddedMessage(r.timestamp, r.role, r.content, emb))
        self._rebuild_index()
        self._prune()

    def add_message(self, role: str, content: str) -> None:  # type: ignore[override]
        super().add_message(role, content)
        from .sql_db import EmbeddingMessage

        Session = self.session_factory
        msg = self._messages[-1]
        with Session() as s:
            s.add(
                EmbeddingMessage(
                    timestamp=msg.timestamp,
                    role=role,
                    content=content,
                    embedding=[float(x) for x in msg.embedding],
                )
            )
            s.commit()
