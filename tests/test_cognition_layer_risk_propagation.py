from vector_service.cognition_layer import CognitionLayer
from vector_metrics_db import VectorMetricsDB


class DummyRetriever:
    def __init__(self, hits):
        self.hits = hits

    def search(self, query, top_k=5, session_id="", **kwargs):
        return list(self.hits)


def test_precomputed_risk_scores_penalise_weights(tmp_path):
    hits = [
        {
            "origin_db": "bot",
            "record_id": "b1",
            "score": 0.5,
            "text": "bot",
            "metadata": {"name": "b", "risk_score": 0.0},
        },
        {
            "origin_db": "workflow",
            "record_id": "w1",
            "score": 0.5,
            "text": "wf",
            "metadata": {"title": "w", "risk_score": 2.0},
        },
    ]
    retriever = DummyRetriever(hits)
    vec_db = VectorMetricsDB(tmp_path / "vec.db")
    layer = CognitionLayer(retriever=retriever, vector_metrics=vec_db)

    _ctx, sid = layer.query("q", top_k=2)
    layer.record_patch_outcome(sid, True, contribution=1.0)

    weights = vec_db.get_db_weights()
    assert weights["workflow"] < weights["bot"]

