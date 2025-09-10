import asyncio
import pytest

from vector_service.embedding_backfill import consume_record_changes, EmbeddingBackfill


def test_event_consumer_batches_concurrent_events(monkeypatch):
    async def _run():
        calls: list[list[str]] = []

        class DummyBus:
            def __init__(self) -> None:
                self.subs: dict[str, list[callable]] = {}

            def subscribe_async(self, topic: str, cb):
                self.subs.setdefault(topic, []).append(cb)

            def publish(self, topic: str, event):
                for cb in self.subs.get(topic, []):
                    asyncio.create_task(cb(topic, event))

        bus = DummyBus()

        def fake_run(self, *, dbs=None, db=None, **_k):
            if dbs:
                calls.append(sorted(dbs))
            elif db:
                calls.append([db])

        monkeypatch.setattr(EmbeddingBackfill, "run", fake_run)

        consumer = asyncio.create_task(
            consume_record_changes(bus=bus, backend="annoy", batch_size=1)
        )

        async def publish(name: str):
            bus.publish("db.record_changed", {"db": name})

        await asyncio.gather(
            *[publish("code") for _ in range(3)],
            *[publish("bot") for _ in range(3)],
        )

        await asyncio.sleep(0.1)
        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer

        processed = {n for call in calls for n in call}
        assert {"code", "bot"} <= processed

    asyncio.run(_run())
