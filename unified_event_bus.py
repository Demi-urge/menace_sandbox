from __future__ import annotations

"""Lightweight publish/subscribe event bus used by Menace bots.

``UnifiedEventBus`` provides an in-process dispatcher for topic-based events.
Callbacks can be registered with :meth:`subscribe` or :meth:`subscribe_async`
and messages are broadcast via :meth:`publish`.  Events may optionally be
persisted to SQLite and mirrored to a :class:`NetworkedEventBus` (RabbitMQ)
when ``rabbitmq_host`` is supplied.

The bus also integrates a small review queue so that costly callbacks may be
audited by :class:`~automated_reviewer.AutomatedReviewer` implementations.
"""

from collections import defaultdict
from threading import Lock
from typing import Callable, Dict, List, Optional, Awaitable, Protocol
import asyncio
import threading
import queue
import sqlite3
import json
import time
import logging

from db_router import GLOBAL_ROUTER

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from .automated_reviewer import AutomatedReviewer
else:
    class AutomatedReviewer(Protocol):  # type: ignore[misc]
        def handle(self, event: object) -> None:
            ...

from .resilience import CircuitBreaker, CircuitOpenError, retry_with_backoff
from .logging_utils import set_correlation_id

logger = logging.getLogger(__name__)


class EventBus(Protocol):
    """Protocol for event bus implementations."""

    def subscribe(
        self, topic: str, callback: Callable[[str, object], None]
    ) -> None:  # pragma: no cover - interface
        ...

    def subscribe_async(
        self, topic: str, callback: Callable[[str, object], Awaitable[None]]
    ) -> None:  # pragma: no cover - interface
        ...

    def publish(
        self, topic: str, event: object
    ) -> None:  # pragma: no cover - interface
        ...


class UnifiedEventBus:
    """In-memory or networked publish/subscribe interface."""

    def __init__(
        self,
        persist_path: Optional[str] = None,
        *,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        rabbitmq_host: Optional[str] = None,
        rethrow_errors: bool = False,
        collect_errors: bool = False,
        reviewer: Optional[AutomatedReviewer] = None,
    ) -> None:
        self._subs: Dict[str, List[Callable[[str, object], None]]] = defaultdict(list)
        self._async_subs: Dict[str, List[Callable[[str, object], Awaitable[None]]]] = (
            defaultdict(list)
        )
        self._lock = Lock()
        self._loop = loop
        if self._loop is None:
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        self._persist: Optional[sqlite3.Connection] = None
        self._network: Optional[EventBus] = None
        self._circuit: CircuitBreaker | None = None
        self._rethrow_errors = rethrow_errors
        self._collect_errors = collect_errors
        self.callback_errors: List[Exception] = []
        self._review_queue: queue.Queue[object] = queue.Queue()
        self._review_stop = threading.Event()
        self._review_thread: Optional[threading.Thread] = None
        self.last_review_event: Optional[object] = None
        self._reviewer = reviewer
        if rabbitmq_host:
            try:
                from .networked_event_bus import NetworkedEventBus

                self._network = NetworkedEventBus(host=rabbitmq_host, loop=self._loop)
                self._circuit = CircuitBreaker()
            except Exception:  # pragma: no cover - optional
                self._network = None
        if persist_path:
            # allow event persistence from multiple threads
            if GLOBAL_ROUTER is None:
                raise RuntimeError("Database router is not initialised")
            self._persist = GLOBAL_ROUTER.get_connection("events")
            self._persist.execute(
                "CREATE TABLE IF NOT EXISTS events(ts REAL, topic TEXT, payload TEXT)"
            )
            self._persist.commit()

    def subscribe(self, topic: str, callback: Callable[[str, object], None]) -> None:
        """Register *callback* to receive events for *topic*."""
        if self._network:
            self._network.subscribe(topic, callback)
            return
        with self._lock:
            self._subs[topic].append(callback)

    def subscribe_async(
        self, topic: str, callback: Callable[[str, object], Awaitable[None]]
    ) -> None:
        """Register an async callback for *topic*."""
        if self._network:
            self._network.subscribe_async(topic, callback)
            return
        with self._lock:
            self._async_subs[topic].append(callback)

    # ------------------------------------------------------------------
    def _ensure_review_consumer(self) -> None:
        if self._review_thread:
            return

        def _consume() -> None:
            while not self._review_stop.is_set():
                try:
                    payload = self._review_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    self._handle_review_event(payload)
                except Exception:
                    logger.exception("review consumer failed")
                finally:
                    self._review_queue.task_done()

        self.subscribe("review:flag", lambda t, e: self._review_queue.put(e))
        self._review_thread = threading.Thread(target=_consume, daemon=True)
        self._review_thread.start()

    def _handle_review_event(self, payload: object) -> None:
        """Default handler for review events."""
        self.last_review_event = payload
        try:
            bot_id = None
            if isinstance(payload, dict):
                bot_id = payload.get("bot_id")
            if self._reviewer:
                try:
                    self._reviewer.handle(payload)
                except Exception:
                    logger.exception("automated reviewer failed")
            logger.info("bot %s flagged for review", bot_id)
        except Exception:
            logger.exception("review handler failed")

    def publish(self, topic: str, event: object) -> None:
        """Send *event* to all subscribers of *topic*."""
        if isinstance(event, dict) and "correlation_id" in event:
            set_correlation_id(str(event.get("correlation_id")))
        if self._network:
            callbacks: List[Callable[[str, object], None]] = []
            async_callbacks: List[Callable[[str, object], Awaitable[None]]] = []
            try:
                assert self._circuit is not None
                retry_with_backoff(
                    lambda: self._circuit.call(lambda: self._network.publish(topic, event)),
                    attempts=3,
                    logger=logger,
                )
            except CircuitOpenError as exc:
                logger.error("event bus circuit open: %s", exc)
                if self._collect_errors:
                    self.callback_errors.append(exc)
                if self._rethrow_errors:
                    raise
            except Exception as exc:  # pragma: no cover - runtime issues
                logger.warning("event publication failed: %s", exc, exc_info=True)
                if self._collect_errors:
                    self.callback_errors.append(exc)
                if self._rethrow_errors:
                    raise
        else:
            with self._lock:
                callbacks = list(self._subs.get(topic, []))
                async_callbacks = list(self._async_subs.get(topic, []))
        if self._persist:
            try:
                self._persist.execute(
                    "INSERT INTO events(ts, topic, payload) VALUES (?,?,?)",
                    (time.time(), topic, json.dumps(event)),
                )
                self._persist.commit()
            except Exception as exc:
                logger.error("failed persisting event", exc_info=True)
                if self._collect_errors:
                    self.callback_errors.append(exc)
                if self._rethrow_errors:
                    raise
        for cb in callbacks:
            try:
                cb(topic, event)
            except Exception as exc:
                logger.error("subscriber failed", exc_info=True)
                if self._collect_errors:
                    self.callback_errors.append(exc)
                if self._rethrow_errors:
                    raise
        for acb in async_callbacks:
            if self._loop:

                async def _run(acb=acb) -> None:
                    try:
                        await acb(topic, event)
                    except Exception as exc:  # pragma: no cover - runtime errors
                        logger.error("async subscriber failed", exc_info=True)
                        if self._collect_errors:
                            self.callback_errors.append(exc)
                        if self._rethrow_errors:
                            raise

                try:
                    self._loop.create_task(_run())
                except Exception as exc:
                    logger.error("scheduling async subscriber failed", exc_info=True)
                    if self._collect_errors:
                        self.callback_errors.append(exc)
                    if self._rethrow_errors:
                        raise
        set_correlation_id(None)

    def close(self) -> None:
        """Close the underlying networked bus if used."""
        if self._network:
            try:
                close = getattr(self._network, "close", None)
                if close:
                    close()
                if self._circuit:
                    self._circuit._failures = 0
                    self._circuit._opened_until = 0.0
            except Exception as exc:  # pragma: no cover - optional
                logger.error("failed closing networked bus", exc_info=True)
                if self._collect_errors:
                    self.callback_errors.append(exc)
                if self._rethrow_errors:
                    raise
        if self._review_thread:
            self._review_stop.set()
            self._review_thread.join(timeout=0.5)

    def replay(self, since: float = 0.0) -> None:
        """Replay persisted events newer than ``since``."""
        if not self._persist:
            return
        cur = self._persist.execute(
            "SELECT ts, topic, payload FROM events WHERE ts>=? ORDER BY ts",
            (since,),
        )
        rows = cur.fetchall()
        for ts, topic, payload in rows:
            try:
                data = json.loads(payload)
            except Exception:
                data = payload
            self.publish(topic, data)

    def flag_for_review(self, bot_id: str, *, severity: str | None = None) -> None:
        """Invoke the automated reviewer for *bot_id* immediately."""
        payload = {"bot_id": bot_id}
        if severity is not None:
            payload["severity"] = severity
        try:
            if self._reviewer:
                self._handle_review_event(payload)
            else:
                # fall back to legacy behaviour when no reviewer is configured
                self._ensure_review_consumer()
                self.publish("review:flag", payload)
        except Exception as exc:
            logger.error("failed flag_for_review", exc_info=True)
            if self._collect_errors:
                self.callback_errors.append(exc)
            if self._rethrow_errors:
                raise


__all__ = ["UnifiedEventBus", "EventBus", "AutomatedReviewer"]
