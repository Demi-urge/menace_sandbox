from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, List, Tuple
import os


@dataclass
class ReactionNode:
    phrase: str
    reaction: str
    timestamp: float
    prev: Optional["ReactionNode"] = None
    next: Optional["ReactionNode"] = None


class ReactionHistory:
    """Doubly linked list storing message-reaction pairs with rapid time decay."""

    def __init__(
        self,
        ttl_seconds: float = 60.0,
        session_factory: Optional[callable] = None,
        *,
        db_url: Optional[str] = None,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.head: Optional[ReactionNode] = None
        self.tail: Optional[ReactionNode] = None
        if session_factory is None:
            from .sql_db import create_session as create_sql_session, ensure_schema

            ensure_schema(db_url or os.environ.get("NEURO_DB_URL", "sqlite://"))
            session_factory = create_sql_session(db_url)
        self.session_factory = session_factory

    def add_pair(
        self,
        phrase: str,
        reaction: str,
        *,
        timestamp: Optional[float] = None,
        archived: bool = False,
    ) -> None:
        """Add a new message-reaction pair to the history."""
        ts = time.time() if timestamp is None else timestamp
        node = ReactionNode(phrase=phrase, reaction=reaction, timestamp=ts)
        if self.tail is None:
            self.head = self.tail = node
        else:
            node.prev = self.tail
            self.tail.next = node
            self.tail = node
        self.prune()
        from .sql_db import log_rl_feedback

        log_rl_feedback(
            phrase,
            reaction,
            session_factory=self.session_factory,
        )

    def prune(self) -> None:
        """Remove entries that have expired based on ttl_seconds."""
        if self.ttl_seconds is None:
            return
        expiry = time.time() - self.ttl_seconds
        current = self.head
        while current and current.timestamp < expiry:
            next_node = current.next
            if next_node:
                next_node.prev = None
            current = next_node
        self.head = current
        if current is None:
            self.tail = None

    def get_pairs(self) -> List[Tuple[str, str]]:
        """Return the stored message-reaction pairs in order."""
        self.prune()
        pairs: List[Tuple[str, str]] = []
        current = self.head
        while current:
            pairs.append((current.phrase, current.reaction))
            current = current.next
        return pairs


class DatabaseReactionHistory:
    """Reaction history persisted to a SQL database."""

    def __init__(
        self,
        user_id: str,
        *,
        ttl_seconds: float = 60.0,
        session_factory: Optional[callable] = None,
        db_url: Optional[str] = None,
    ) -> None:
        self.user_id = user_id
        self.ttl_seconds = ttl_seconds
        if session_factory is None:
            from .sql_db import create_session as create_sql_session, ensure_schema

            ensure_schema(db_url or os.environ.get("NEURO_DB_URL", "sqlite://"))
            session_factory = create_sql_session(db_url)
        self.session_factory = session_factory

    def add_pair(
        self,
        phrase: str,
        reaction: str,
        *,
        timestamp: Optional[float] = None,
        archived: bool = False,
    ) -> None:
        ts = time.time() if timestamp is None else timestamp
        from .sql_db import ReactionPair

        Session = self.session_factory
        with Session() as s:
            s.add(
                ReactionPair(
                    user_id=self.user_id,
                    phrase=phrase,
                    reaction=reaction,
                    timestamp=ts,
                    archived=archived,
                )
            )
            s.commit()

    def get_pairs(self) -> List[Tuple[str, str]]:
        from .sql_db import ReactionPair

        Session = self.session_factory
        with Session() as s:
            rows = (
                s.query(ReactionPair)
                .filter_by(user_id=self.user_id, archived=False)
                .order_by(ReactionPair.timestamp)
                .all()
            )

        if self.ttl_seconds is not None:
            expiry = time.time() - self.ttl_seconds
            rows = [r for r in rows if r.timestamp >= expiry]

        return [(r.phrase, r.reaction) for r in rows]
