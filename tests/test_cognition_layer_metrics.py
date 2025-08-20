import types
import time

from vector_service.context_builder import ContextBuilder
from vector_service.cognition_layer import CognitionLayer
from vector_metrics_db import VectorMetricsDB


class DummyRetriever:
    def search(self, query, top_k=5, session_id=""):
        time.sleep(0.001)
        return [
            {
                "origin_db": "bot",
                "record_id": 1,
                "score": 0.9,
                "metadata": {"name": "alpha", "timestamp": time.time() - 30.0},
            }
        ]


def test_retrieval_metrics_persist(tmp_path):
    db = VectorMetricsDB(tmp_path / "vec.db")
    builder = ContextBuilder(retriever=DummyRetriever())
    layer = CognitionLayer(
        context_builder=builder,
        vector_metrics=db,
        patch_logger=types.SimpleNamespace(
            track_contributors=lambda *a, **k: None,
            track_contributors_async=lambda *a, **k: None,
        ),
    )

    layer.query("hello world")

    rows = db.conn.execute(
        "SELECT tokens, wall_time_ms, prompt_tokens, age FROM vector_metrics WHERE event_type='retrieval'"
    ).fetchall()
    assert rows
    for tokens, wall_ms, prompt_tokens, age in rows:
        assert tokens > 0
        assert wall_ms > 0.0
        assert prompt_tokens > 0
        assert age >= 0
