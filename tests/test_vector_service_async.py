import asyncio
import json

import vector_service.decorators as dec
from vector_service.decorators import log_and_measure
from vector_service.retriever import Retriever
from vector_service.context_builder import ContextBuilder
from vector_service.patch_logger import PatchLogger


class Gauge:
    def __init__(self):
        self.inc_calls = 0
        self.set_calls = []

    def labels(self, *args):
        return self

    def inc(self):
        self.inc_calls += 1

    def set(self, value):
        self.set_calls.append(value)


def test_log_and_measure_async(monkeypatch):
    g1, g2, g3 = Gauge(), Gauge(), Gauge()
    monkeypatch.setattr(dec, "_CALL_COUNT", g1)
    monkeypatch.setattr(dec, "_LATENCY_GAUGE", g2)
    monkeypatch.setattr(dec, "_RESULT_SIZE_GAUGE", g3)

    @log_and_measure
    async def example(n: int):
        await asyncio.sleep(0)
        return list(range(n))

    res = asyncio.run(example(3))
    assert res == [0, 1, 2]
    assert g1.inc_calls == 1
    assert g3.set_calls == [3]


def test_retriever_search_async(monkeypatch):
    g1, g2, g3 = Gauge(), Gauge(), Gauge()
    monkeypatch.setattr(dec, "_CALL_COUNT", g1)
    monkeypatch.setattr(dec, "_LATENCY_GAUGE", g2)
    monkeypatch.setattr(dec, "_RESULT_SIZE_GAUGE", g3)

    class DummyResult:
        def __init__(self):
            self.record_id = 1
            self.origin_db = "error"
            self.score = 0.5
            self.metadata = {"message": "oops", "redacted": True}

        def to_dict(self):
            return {
                "origin_db": self.origin_db,
                "record_id": self.record_id,
                "score": self.score,
                "metadata": self.metadata,
            }

    class DummyUR:
        def retrieve_with_confidence(self, query, top_k=5):
            return [DummyResult()], 0.9, None

    r = Retriever(retriever=DummyUR())
    result = asyncio.run(r.search_async("test", session_id="s"))
    assert result and result[0]["record_id"] == 1
    assert g1.inc_calls == 1
    assert g3.set_calls == [1]


def test_context_builder_build_async(monkeypatch):
    class DummyRetriever:
        def search(self, query, top_k=5, session_id=None, **kwargs):
            return [
                {
                    "origin_db": "error",
                    "record_id": 1,
                    "score": 0.5,
                    "metadata": {"message": "fail", "redacted": True},
                }
            ]

    g1, g2, g3 = Gauge(), Gauge(), Gauge()
    monkeypatch.setattr(dec, "_CALL_COUNT", g1)
    monkeypatch.setattr(dec, "_LATENCY_GAUGE", g2)
    monkeypatch.setattr(dec, "_RESULT_SIZE_GAUGE", g3)

    builder = ContextBuilder(retriever=DummyRetriever())
    ctx = asyncio.run(builder.build_async("query"))
    data = json.loads(ctx)
    assert "errors" in data and data["errors"][0]["id"] == 1
    assert g1.inc_calls == 2
    assert g3.set_calls[1:] == [len(ctx)]


def test_patch_logger_track_contributors_async(monkeypatch):
    g1, g2, g3 = Gauge(), Gauge(), Gauge()
    monkeypatch.setattr(dec, "_CALL_COUNT", g1)
    monkeypatch.setattr(dec, "_LATENCY_GAUGE", g2)
    monkeypatch.setattr(dec, "_RESULT_SIZE_GAUGE", g3)

    class DummyMetricsDB:
        def __init__(self):
            self.logged = []

        def log_patch_outcome(self, patch_id, result, pairs, session_id=""):
            self.logged.append((patch_id, result, pairs, session_id))

    db = DummyMetricsDB()
    pl = PatchLogger(metrics_db=db)
    asyncio.run(pl.track_contributors_async(["err:1", "code:2"], True, patch_id="99", session_id="s"))
    assert db.logged and db.logged[0][0] == "99"
    assert g1.inc_calls == 1
