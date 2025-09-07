import json
import pytest

from vector_service.context_builder import ContextBuilder
from vector_service.cognition_layer import CognitionLayer
from vector_metrics_db import VectorMetricsDB


class DummyRetriever:
    def __init__(self, hits):
        self.hits = hits

    def search(self, query: str, top_k: int = 5, session_id: str = "", **_):
        return list(self.hits)


def test_db_weights_shift_priorities(tmp_path):
    hits = [
        {
            "origin_db": "bot",
            "record_id": "b1",
            "score": 1.0,
            "text": "bot",
            "metadata": {"name": "b"},
        },
        {
            "origin_db": "workflow",
            "record_id": "w1",
            "score": 1.0,
            "text": "wf",
            "metadata": {"title": "w"},
        },
    ]
    retriever = DummyRetriever(hits)
    vec_db = VectorMetricsDB(tmp_path / "vec.db")
    builder = ContextBuilder(retriever=retriever)
    builder.refresh_db_weights(vector_metrics=vec_db)
    _ctx, _sid, vectors = builder.build_context("q", include_vectors=True, top_k=1)
    scores = {o: s for o, _vid, s in vectors}
    assert scores["bot"] == pytest.approx(scores["workflow"])

    vec_db.update_db_weight("workflow", 1.0)
    vec_db.update_db_weight("bot", -0.5)
    builder.refresh_db_weights(vector_metrics=vec_db)
    _ctx2, _sid2, vectors2 = builder.build_context("q", include_vectors=True, top_k=1)
    scores2 = {o: s for o, _vid, s in vectors2}
    assert scores2["workflow"] > scores2["bot"]


def test_risk_scores_penalize_and_propagate(tmp_path):
    hits = [
        {
            "origin_db": "bot",
            "record_id": "b1",
            "score": 0.5,
            "text": "bot",
            "metadata": {"name": "b", "roi": 1.0, "risk_score": 0.0},
        },
        {
            "origin_db": "workflow",
            "record_id": "w1",
            "score": 0.5,
            "text": "wf",
            "metadata": {"title": "w", "roi": 1.0, "risk_score": 2.0},
        },
    ]
    retriever = DummyRetriever(hits)
    vec_db = VectorMetricsDB(tmp_path / "vec.db")
    builder = ContextBuilder(retriever=retriever)
    layer = CognitionLayer(
        retriever=retriever, vector_metrics=vec_db, context_builder=builder
    )
    ctx, sid = layer.query("q", top_k=1)
    parsed = json.loads(ctx)
    assert parsed["bots"][0]["roi"] == 1.0
    assert parsed["workflows"][0]["risk_score"] == 2.0
    scores = {o: s for o, _vid, s in layer._session_vectors[sid]}
    assert scores["bot"] > scores["workflow"]
    meta = layer._retrieval_meta[sid]
    assert meta["bot:b1"]["roi"] == 1.0
    assert meta["workflow:w1"]["risk_score"] == 2.0

    retriever.hits[0]["metadata"]["risk_score"] = 3.0
    retriever.hits[1]["metadata"]["risk_score"] = 0.0
    ctx2, sid2 = layer.query("q", top_k=1)
    scores2 = {o: s for o, _vid, s in layer._session_vectors[sid2]}
    assert scores2["workflow"] > scores2["bot"]
    parsed2 = json.loads(ctx2)
    assert parsed2["workflows"][0]["risk_score"] == 0.0
    assert parsed2["bots"][0]["risk_score"] == 3.0
