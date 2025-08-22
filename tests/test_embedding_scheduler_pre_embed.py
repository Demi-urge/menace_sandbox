import pytest

from vector_service.embedding_scheduler import EmbeddingScheduler

class DummyBus:
    def subscribe(self, topic, handler):
        self.handler = handler

class DummyPatchSafety:
    def __init__(self):
        self.checked = []
    def pre_embed_check(self, meta):
        self.checked.append(meta)
        return meta.get("ok", True)


def test_pre_embed_check_filters(monkeypatch):
    bus = DummyBus()
    ps = DummyPatchSafety()
    sched = EmbeddingScheduler(event_bus=bus, patch_safety=ps)
    # failing metadata should be skipped
    bus.handler("db:new_record", {"source": "bot", "metadata": {"ok": False}})
    assert ps.checked == [{"ok": False}]
    assert sched._event_counts["bot"] == 0
    # passing metadata increments count
    bus.handler("db:new_record", {"source": "bot", "metadata": {"ok": True}})
    assert sched._event_counts["bot"] == 1
