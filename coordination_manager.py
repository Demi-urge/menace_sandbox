"""Inter-bot communication and coordination utilities."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from db_router import GLOBAL_ROUTER, init_db_router

try:
    import pika  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pika = None  # type: ignore


@dataclass
class Message:
    """Standardised message schema for bot communication."""

    sender: str
    recipient: str
    task: str
    payload: str
    ts: str = datetime.utcnow().isoformat()


class MessageLog:
    """SQLite-backed log of exchanged messages."""

    def __init__(self, path: Path | str = "messages.db") -> None:
        router = GLOBAL_ROUTER or init_db_router("coordination_manager", local_db_path=str(path))
        self.conn = router.get_connection("messages")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages(
                sender TEXT,
                recipient TEXT,
                task TEXT,
                payload TEXT,
                ts TEXT
            )
            """
        )
        self.conn.commit()

    def add(self, msg: Message) -> None:
        self.conn.execute(
            "INSERT INTO messages(sender, recipient, task, payload, ts) VALUES(?,?,?,?,?)",
            (msg.sender, msg.recipient, msg.task, msg.payload, msg.ts),
        )
        self.conn.commit()

    def fetch(self) -> List[Tuple[str, str, str, str, str]]:
        cur = self.conn.execute(
            "SELECT sender, recipient, task, payload, ts FROM messages"
        )
        return cur.fetchall()


class CoordinationManager:
    """Central coordination layer using RabbitMQ if available."""

    def __init__(self, queue_url: str = "amqp://guest:guest@localhost/", log: MessageLog | None = None) -> None:
        self.log = log or MessageLog()
        self._use_pika = False
        if pika:
            try:
                self.conn = pika.BlockingConnection(pika.URLParameters(queue_url))
                self.ch = self.conn.channel()
                self.ch.queue_declare(queue="tasks")
                self._use_pika = True
            except Exception:  # pragma: no cover - no RabbitMQ
                self.conn = None
                self.ch = None
        if not self._use_pika:
            from queue import Queue

            self.queue: Queue[str] = Queue()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("CoordinationManager")

    def send(self, msg: Message) -> None:
        body = json.dumps(asdict(msg))
        if self._use_pika:
            assert self.ch is not None
            self.ch.basic_publish(exchange="", routing_key="tasks", body=body)
        else:  # pragma: no cover - fallback
            self.queue.put(body)
        self.log.add(msg)

    def receive(self) -> Message | None:
        if self._use_pika:
            assert self.ch is not None
            method, _, body = self.ch.basic_get("tasks", auto_ack=True)
            if not body:
                return None
        else:  # pragma: no cover - fallback
            if self.queue.empty():
                return None
            body = self.queue.get().encode()
        data = json.loads(body)
        return Message(**data)


class TaskDistributor:
    """Distribute tasks across bots using simple round-robin."""

    def __init__(self) -> None:
        self.index = 0

    def assign(self, tasks: Iterable[str], bots: Iterable[str]) -> List[Tuple[str, str]]:
        bots_list = list(bots)
        if not bots_list:
            return []
        assignments: List[Tuple[str, str]] = []
        for t in tasks:
            bot = bots_list[self.index % len(bots_list)]
            assignments.append((bot, t))
            self.index += 1
        return assignments


__all__ = [
    "Message",
    "MessageLog",
    "CoordinationManager",
    "TaskDistributor",
]
