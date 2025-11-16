from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional
import os

@dataclass
class Message:
    timestamp: float
    role: str
    content: str

@dataclass
class ConversationMemory:
    """Simple memory manager for conversations."""

    max_messages: int = 6
    ttl_seconds: Optional[int] = None
    queue: deque[Message] = field(default_factory=deque)
    stack: List[Message] = field(default_factory=list)

    def add_message(self, role: str, content: str) -> None:
        now = time.time()
        msg = Message(timestamp=now, role=role, content=content)
        self.queue.append(msg)
        if len(self.queue) > self.max_messages:
            self.queue.popleft()
        self.stack.append(msg)
        self.prune()

    def prune(self) -> None:
        """Remove messages that exceed the TTL."""
        if self.ttl_seconds is None:
            return
        expiry = time.time() - self.ttl_seconds
        while self.queue and self.queue[0].timestamp < expiry:
            self.queue.popleft()
        self.stack = [m for m in self.stack if m.timestamp >= expiry]

    def get_recent_messages(self) -> List[Message]:
        """Return a list of recent messages."""
        return list(self.queue)

    def push_stack(self, role: str, content: str) -> None:
        """Push a message onto the stack."""
        self.stack.append(Message(timestamp=time.time(), role=role, content=content))
        self.prune()

    def pop_stack(self) -> Optional[Message]:
        """Pop a message from the stack if it exists."""
        if self.stack:
            return self.stack.pop()
        return None

@dataclass
class DatabaseConversationMemory(ConversationMemory):
    """Persistent conversation memory backed by a SQL database."""

    user_id: str = ""
    session_factory: Optional[callable] = None
    db_url: Optional[str] = None

    def __post_init__(self) -> None:
        if self.session_factory is None:
            from .sql_db import create_session as create_sql_session, ensure_schema

            ensure_schema(self.db_url or os.environ.get("NEURO_DB_URL", "sqlite://"))
            self.session_factory = create_sql_session(self.db_url)
        # load existing history
        self.queue = deque()
        self.stack = []
        Session = self.session_factory
        from .sql_db import ConversationMessage
        with Session() as s:
            rows = (
                s.query(ConversationMessage)
                .filter_by(user_id=self.user_id)
                .order_by(ConversationMessage.timestamp.desc())
                .limit(self.max_messages)
                .all()
            )
        for r in reversed(rows):
            msg = Message(timestamp=r.timestamp, role=r.role, content=r.message)
            self.queue.append(msg)
            self.stack.append(msg)
        self.prune()

    def add_message(self, role: str, content: str) -> None:
        super().add_message(role, content)
        Session = self.session_factory
        from .sql_db import ConversationMessage
        msg = self.stack[-1]
        with Session() as s:
            rec = ConversationMessage(
                user_id=self.user_id,
                role=role,
                message=content,
                timestamp=msg.timestamp,
            )
            s.add(rec)
            s.commit()

