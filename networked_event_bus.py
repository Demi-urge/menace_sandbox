from __future__ import annotations

"""RabbitMQ backed event bus for distributed bots.

``NetworkedEventBus`` implements the same API as
:class:`~unified_event_bus.UnifiedEventBus` but transports messages over
RabbitMQ using :mod:`pika`.  When the broker or dependency is unavailable it
falls back to an in-process queue so tests and single-node deployments can
continue to operate.

The public surface mirrors ``subscribe``, ``subscribe_async`` and ``publish``
methods so components can switch between local and networked buses with
minimal changes.
"""

from typing import Callable, Awaitable, Dict, List, Optional
import json
import asyncio
import threading
import logging
import queue

from .resilience import (
    CircuitBreaker,
    CircuitOpenError,
    PublishError,
    retry_with_backoff,
)
from .logging_utils import set_correlation_id

try:
    import pika
except Exception:  # pragma: no cover - optional
    pika = None  # type: ignore
    logging.getLogger(__name__).warning(
        "pika not available - using in-process event bus"
    )


class NetworkedEventBus:
    """Simple RabbitMQ based publish/subscribe bus."""

    def __init__(
        self,
        host: str = "localhost",
        *,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        self._loop = loop
        if self._loop is None:
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        self._host = host
        self._subs: Dict[str, List[Callable[[str, object], None]]] = {}
        self._async_subs: Dict[str, List[Callable[[str, object], Awaitable[None]]]] = {}
        self._consumers: Dict[str, threading.Thread] = {}
        self._closing = threading.Event()
        self._circuit = CircuitBreaker()
        if pika:
            self._conn = pika.BlockingConnection(pika.ConnectionParameters(host=host))
            self._channel = self._conn.channel()
            self._queues = None
        else:
            self._conn = None
            self._channel = None
            self._queues: Dict[str, queue.Queue[object]] = {}

    # ------------------------------------------------------------------
    def subscribe(self, topic: str, callback: Callable[[str, object], None]) -> None:
        if self._channel:
            self._channel.queue_declare(queue=topic, durable=True)
        self._subs.setdefault(topic, []).append(callback)
        self._ensure_consumer(topic)

    def subscribe_async(
        self, topic: str, callback: Callable[[str, object], Awaitable[None]]
    ) -> None:
        if self._channel:
            self._channel.queue_declare(queue=topic, durable=True)
        self._async_subs.setdefault(topic, []).append(callback)
        self._ensure_consumer(topic)

    # ------------------------------------------------------------------
    def _ensure_consumer(self, topic: str) -> None:
        if topic in self._consumers:
            return
        if self._channel:

            def _consume() -> None:
                conn = pika.BlockingConnection(
                    pika.ConnectionParameters(host=self._host)
                )
                channel = conn.channel()
                channel.queue_declare(queue=topic, durable=True)
                for method_frame, properties, body in channel.consume(
                    queue=topic, inactivity_timeout=1
                ):
                    if self._closing.is_set():
                        break
                    if body is None:
                        continue
                    data = json.loads(body.decode("utf-8"))
                    cid = None
                    if isinstance(data, dict):
                        cid = data.get("correlation_id")
                    set_correlation_id(str(cid) if cid is not None else None)
                    for cb in self._subs.get(topic, []):
                        try:
                            cb(topic, data)
                        except Exception as exc:
                            logging.getLogger(__name__).error(
                                "subscriber failed: %s", exc
                            )
                    for acb in self._async_subs.get(topic, []):
                        try:
                            self._loop.call_soon_threadsafe(
                                lambda: self._loop.create_task(acb(topic, data))
                            )
                        except Exception as exc:
                            logging.getLogger(__name__).error(
                                "async subscriber failed: %s", exc
                            )
                    set_correlation_id(None)
                    channel.basic_ack(method_frame.delivery_tag)
                try:
                    channel.close()
                except Exception as exc:
                    logging.getLogger(__name__).error(
                        "consumer channel close failed: %s", exc
                    )
                try:
                    conn.close()
                except Exception as exc:
                    logging.getLogger(__name__).error(
                        "consumer connection close failed: %s", exc
                    )

        else:
            q = self._queues.setdefault(topic, queue.Queue())

            def _consume() -> None:
                while not self._closing.is_set():
                    try:
                        data = q.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    cid = None
                    if isinstance(data, dict):
                        cid = data.get("correlation_id")
                    set_correlation_id(str(cid) if cid is not None else None)
                    for cb in self._subs.get(topic, []):
                        try:
                            cb(topic, data)
                        except Exception as exc:
                            logging.getLogger(__name__).error(
                                "subscriber failed: %s", exc
                            )
                    for acb in self._async_subs.get(topic, []):
                        try:
                            self._loop.call_soon_threadsafe(
                                lambda: self._loop.create_task(acb(topic, data))
                            )
                        except Exception as exc:
                            logging.getLogger(__name__).error(
                                "async subscriber failed: %s", exc
                            )
                    set_correlation_id(None)
                    q.task_done()

        thread = threading.Thread(target=_consume, daemon=True)
        self._consumers[topic] = thread
        thread.start()

    # ------------------------------------------------------------------
    def publish(self, topic: str, event: object) -> None:
        if isinstance(event, dict):
            cid = event.get("correlation_id")
            set_correlation_id(str(cid) if cid is not None else None)
        logger = logging.getLogger(__name__)
        if self._channel:
            def _send() -> None:
                self._channel.queue_declare(queue=topic, durable=True)
                self._channel.basic_publish(
                    exchange="",
                    routing_key=topic,
                    body=json.dumps(event).encode("utf-8"),
                    properties=pika.BasicProperties(delivery_mode=2),
                )

            try:
                retry_with_backoff(
                    lambda: self._circuit.call(_send),
                    attempts=3,
                    logger=logger,
                )
            except CircuitOpenError as exc:
                logger.error("circuit open during publish: %s", exc)
                raise PublishError("circuit open") from exc
            except Exception as exc:
                logger.warning("publish failed after retries: %s", exc, exc_info=True)
                raise PublishError(str(exc)) from exc
        else:
            q = self._queues.setdefault(topic, queue.Queue())
            try:
                q.put(event)
            except Exception as exc:
                logger.warning("publish failed: %s", exc, exc_info=True)
                raise PublishError(str(exc)) from exc
        set_correlation_id(None)

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Stop consumers and close the underlying connection."""
        self._closing.set()
        for thread in list(self._consumers.values()):
            thread.join(timeout=2)
        if not self._channel:
            self._queues.clear()
        if self._channel:
            try:
                self._channel.close()
            except Exception as exc:
                import logging

                logging.getLogger(__name__).error("channel close failed: %s", exc)
        if self._conn:
            try:
                self._conn.close()
            except Exception as exc:
                import logging

                logging.getLogger(__name__).error("connection close failed: %s", exc)


__all__ = ["NetworkedEventBus", "PublishError"]
