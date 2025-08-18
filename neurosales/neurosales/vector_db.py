from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Any
import os
import logging

try:
    import faiss  # type: ignore
    from sentence_transformers import SentenceTransformer
    import numpy as np
except Exception:  # pragma: no cover - optional heavy deps
    faiss = None  # type: ignore
    SentenceTransformer = None  # type: ignore
    np = None  # type: ignore

from analysis.semantic_diff_filter import find_semantic_risks
from governed_embeddings import governed_embed
from .external_integrations import PineconeLogger

logger = logging.getLogger(__name__)


@dataclass
class VectorMessage:
    timestamp: float
    role: str
    content: str
    embedding: List[float]
    priority: float = 1.0
    synced: bool = False


class VectorDB:
    """Hybrid FAISS + Pinecone memory with decay and syncing."""

    def __init__(
        self,
        max_messages: int = 10,
        ttl_seconds: Optional[int] = None,
        decay_factor: float = 0.99,
        *,
        pinecone_index: Optional[str] = None,
        pinecone_key: Optional[str] = None,
        pinecone_env: Optional[str] = None,
        sync_interval: int = 5,
    ) -> None:
        self.max_messages = max_messages
        self.ttl_seconds = ttl_seconds
        self.decay_factor = decay_factor
        self.sync_interval = sync_interval
        self._messages: Deque[VectorMessage] = deque()
        self._unsynced: List[VectorMessage] = []
        self._model = None
        self._index = None
        if SentenceTransformer is not None and faiss is not None:
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            dim = self._model.get_sentence_embedding_dimension()
            self._index = faiss.IndexFlatL2(dim)
        self._pinecone = PineconeLogger(
            pinecone_index,
            api_key=pinecone_key,
            environment=pinecone_env,
        )

    # ------------------------------------------------------------------
    def _decay_priorities(self) -> None:
        for msg in self._messages:
            msg.priority *= self.decay_factor

    # ------------------------------------------------------------------
    def _prune(self) -> None:
        if self.ttl_seconds is None:
            return
        cutoff = time.time() - self.ttl_seconds
        while self._messages and self._messages[0].timestamp < cutoff:
            self._messages.popleft()
        self._rebuild_index()

    # ------------------------------------------------------------------
    def _rebuild_index(self) -> None:
        if self._index is None or np is None:
            return
        self._index.reset()
        if not self._messages:
            return
        embeddings = np.stack([m.embedding for m in self._messages]).astype("float32")
        self._index.add(embeddings)

    # ------------------------------------------------------------------
    def add_message(self, role: str, content: str) -> None:
        if self._model is not None and np is not None:
            alerts = find_semantic_risks(content.splitlines())
            if alerts:
                logger.warning("semantic risks detected: %s", [a[1] for a in alerts])
                return
            vec = governed_embed(content, self._model)
            if vec is None:
                return
            emb_list = np.array(vec, dtype="float32").tolist()
        else:
            emb_list = []
        msg = VectorMessage(time.time(), role, content, emb_list)
        self._messages.append(msg)
        self._unsynced.append(msg)
        if len(self._messages) > self.max_messages:
            self._messages.popleft()
        self._decay_priorities()
        self._prune()
        self._rebuild_index()
        if len(self._unsynced) >= self.sync_interval:
            self.sync()

    # ------------------------------------------------------------------
    def get_recent_messages(self) -> List[VectorMessage]:
        return list(self._messages)

    # ------------------------------------------------------------------
    def most_similar(self, text: str, top_k: int = 3) -> List[VectorMessage]:
        if self._index is None or self._model is None or np is None:
            return []
        if not self._messages:
            return []
        alerts = find_semantic_risks(text.splitlines())
        if alerts:
            logger.warning("semantic risks detected: %s", [a[1] for a in alerts])
            return []
        vec = governed_embed(text, self._model)
        if vec is None:
            return []
        query = np.array(vec, dtype="float32")
        D, I = self._index.search(query.reshape(1, -1), min(top_k, len(self._messages)))
        return [self._messages[i] for i in I[0]]

    # ------------------------------------------------------------------
    def sync(self) -> None:
        if not self._unsynced:
            return
        if self._pinecone is None:
            self._unsynced.clear()
            return
        for msg in list(self._unsynced):
            self._pinecone.log(msg.role, msg.embedding, msg.content)
            msg.synced = True
        self._unsynced.clear()


class DatabaseVectorDB(VectorDB):
    """Vector DB that persists messages to a SQL database."""

    def __init__(
        self,
        *,
        session_factory: Optional[callable] = None,
        db_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if session_factory is None:
            from .sql_db import create_session as create_sql_session, ensure_schema

            ensure_schema(db_url or os.environ.get("NEURO_DB_URL", "sqlite://"))
            session_factory = create_sql_session(db_url)
        self.session_factory = session_factory

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
            self._messages.append(
                VectorMessage(
                    r.timestamp,
                    r.role,
                    r.content,
                    r.embedding or [],
                    r.priority or 1.0,
                    bool(r.synced),
                )
            )
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
                    role=msg.role,
                    content=msg.content,
                    embedding=msg.embedding,
                    priority=msg.priority,
                    synced=msg.synced,
                )
            )
            s.commit()

    def sync(self) -> None:  # type: ignore[override]
        pending = list(self._unsynced)
        super().sync()
        if not pending:
            return
        from .sql_db import EmbeddingMessage

        Session = self.session_factory
        with Session() as s:
            for msg in pending:
                s.query(EmbeddingMessage).filter_by(
                    timestamp=msg.timestamp,
                    role=msg.role,
                    content=msg.content,
                ).update({"synced": True})
            s.commit()


__all__ = ["VectorDB", "VectorMessage", "DatabaseVectorDB"]
