import json
from types import MethodType

from discrepancy_db import DiscrepancyDB, DiscrepancyRecord
from db_router import DBRouter
from vector_metrics_db import VectorMetricsDB
from vector_service.context_builder import ContextBuilder
from vector_service.cognition_layer import CognitionLayer


class DummyPatchLogger:
    def __init__(self):
        self.roi_tracker = None


class DiscrepancyRetriever:
    def __init__(self, db: DiscrepancyDB):
        self.db = db

    def search(self, query: str, top_k: int = 5, session_id: str = "", **_):
        vec = self.db.encode_text(query)
        results = []
        for rid, dist in self.db.search_by_vector(vec, top_k=top_k):
            rec = self.db.get(int(rid))
            results.append(
                {
                    "origin_db": "discrepancy",
                    "record_id": int(rid),
                    "score": 1.0 - float(dist),
                    "metadata": {"message": rec.message, **rec.metadata},
                    "text": rec.message,
                }
            )
        return results


def test_cognition_layer_returns_discrepancies(tmp_path):
    shared_db = tmp_path / "shared.db"
    router = DBRouter("one", str(tmp_path / "local.db"), str(shared_db))
    db = DiscrepancyDB(router=router, vector_index_path=tmp_path / "d.index")

    def fake_encode(self, text: str):
        return [float(len(text))]

    db.encode_text = MethodType(fake_encode, db)

    rec = DiscrepancyRecord(message="unexpected output", metadata={"kind": "test"})
    db.add(rec)

    retriever = DiscrepancyRetriever(db)
    builder = ContextBuilder(retriever=retriever)
    layer = CognitionLayer(
        retriever=retriever,
        context_builder=builder,
        patch_logger=DummyPatchLogger(),
        vector_metrics=VectorMetricsDB(":memory:"),
        roi_tracker=None,
    )

    ctx, sid = layer.query("unexpected output")
    data = json.loads(ctx)
    assert "discrepancies" in data and data["discrepancies"]
    assert any(origin == "discrepancy" for origin, _vid, _s in layer._session_vectors[sid])
    router.close()
