from __future__ import annotations

"""RabbitMQ backed event bus for distributed bots."""

from typing import Callable, Awaitable, Dict, List, Optional
import json
import asyncio
import threading
import logging
import queue

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
                    q.task_done()

        thread = threading.Thread(target=_consume, daemon=True)
        self._consumers[topic] = thread
        thread.start()

    # ------------------------------------------------------------------
    def publish(self, topic: str, event: object) -> None:
        if self._channel:
            try:
                self._channel.queue_declare(queue=topic, durable=True)
                self._channel.basic_publish(
                    exchange="",
                    routing_key=topic,
                    body=json.dumps(event).encode("utf-8"),
                    properties=pika.BasicProperties(delivery_mode=2),
                )
            except Exception as exc:
                logging.getLogger(__name__).warning(
                    "publish failed: %s", exc, exc_info=True
                )
        else:
            q = self._queues.setdefault(topic, queue.Queue())
            try:
                q.put(event)
            except Exception as exc:
                logging.getLogger(__name__).warning(
                    "publish failed: %s", exc, exc_info=True
                )

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


__all__ = ["NetworkedEventBus"]
