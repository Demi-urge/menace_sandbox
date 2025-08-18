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

try:
    from analysis.semantic_diff_filter import find_semantic_risks
except Exception:  # pragma: no cover - optional dependency
    def find_semantic_risks(lines, threshold: float = 0.5):  # type: ignore[override]
        return []
try:
    from compliance.license_fingerprint import check as license_check
except Exception:  # pragma: no cover - optional dependency
    def license_check(text: str):  # type: ignore
        return None
try:
    from security.secret_redactor import redact
except Exception:  # pragma: no cover - optional dependency
    def redact(text: str):  # type: ignore
        return text
import logging
from governed_embeddings import governed_embed

logger = logging.getLogger(__name__)


@dataclass
class EmbeddedMessage:
    timestamp: float
    role: str
    content: str
    embedding: List[float]
    alerts: Optional[List[tuple[str, str, float]]] = None


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
        """Add a message to memory, computing an embedding when safe.

        The text is redacted and checked for licensing issues before being
        scanned with :func:`find_semantic_risks`. If a license is detected the
        message is dropped. If semantic risks are detected the message is
        stored without an embedding and the risk list is attached to the
        ``alerts`` field. Otherwise the message is embedded and indexed for
        similarity search.
        """

        original = redact(content.strip())
        if not original:
            return
        lic = license_check(original)
        if lic:
            logger.warning("license detected: %s", lic)
            return
        alerts = find_semantic_risks(original.splitlines())
        if alerts:
            logger.warning("semantic risks detected: %s", [a[1] for a in alerts])
            msg = EmbeddedMessage(time.time(), role, original, [], alerts)
            self._messages.append(msg)
            if len(self._messages) > self.max_messages:
                self._messages.popleft()
            self._prune()
            return

        if SentenceTransformer is None or faiss is None or np is None:
            msg = EmbeddedMessage(time.time(), role, original, [])
            self._messages.append(msg)
            if len(self._messages) > self.max_messages:
                self._messages.popleft()
            self._prune()
            return

        vec = governed_embed(original, self._model)
        emb_list = np.array(vec, dtype="float32").tolist() if vec is not None else []
        msg = EmbeddedMessage(time.time(), role, original, emb_list)
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
        alerts = find_semantic_risks(text.splitlines())
        if alerts:
            logger.warning("semantic risks detected: %s", [a[1] for a in alerts])
            results: List[EmbeddedMessage] = []
            for m in self._messages:
                msg_alerts = find_semantic_risks(m.content.splitlines())
                if msg_alerts:
                    results.append(
                        EmbeddedMessage(m.timestamp, m.role, m.content, m.embedding, msg_alerts)
                    )
                if len(results) >= top_k:
                    break
            return results
        vec = governed_embed(text, self._model)
        if vec is None:
            return []
        query = np.array(vec, dtype="float32")
        D, indices = self._index.search(
            query.reshape(1, -1), min(top_k, len(self._messages))
        )
        results: List[EmbeddedMessage] = []
        for i in indices[0]:
            msg = self._messages[i]
            msg_alerts = find_semantic_risks(msg.content.splitlines())
            if msg_alerts:
                msg = EmbeddedMessage(
                    msg.timestamp, msg.role, msg.content, msg.embedding, msg_alerts
                )
            results.append(msg)
        return results


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
        before = len(self._messages)
        super().add_message(role, content)
        if len(self._messages) == before:
            return
        msg = self._messages[-1]
        if msg.alerts:
            return

        from .sql_db import EmbeddingMessage

        Session = self.session_factory
        with Session() as s:
            s.add(
                EmbeddingMessage(
                    timestamp=msg.timestamp,
                    role=role,
                    content=msg.content,
                    embedding=[float(x) for x in msg.embedding],
                )
            )
            s.commit()
