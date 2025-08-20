from types import MethodType, SimpleNamespace

import pytest

import menace.failure_learning_system as fls
from vector_service.retriever import Retriever
from vector_service.context_builder import ContextBuilder


@pytest.mark.parametrize("backend", ["annoy", "faiss"])
def test_discrepancy_retrieval_via_context_builder(tmp_path, backend):
    if backend == "faiss":
        pytest.importorskip("faiss")
        pytest.importorskip("numpy")

    db = fls.DiscrepancyDB(
        path=tmp_path / "d.db",
        vector_backend=backend,
        vector_index_path=tmp_path / f"d.{backend}.index",
    )

    def fake_embed(self, text: str):
        return [1.0, 0.0] if "alpha" in text else [0.0, 1.0]

    db._embed = MethodType(fake_embed, db)

    db.log_detection("rule", 0.5, "alpha failure", "wf")
    db.backfill_embeddings()
    rec_id = db.conn.execute(
        "SELECT rowid FROM detections WHERE message='alpha failure'"
    ).fetchone()[0]
    assert str(rec_id) in db._metadata

    meta = db._metadata[str(rec_id)].copy()
    meta["message"] = "alpha failure"

    hit = SimpleNamespace(
        origin_db="discrepancy",
        record_id=rec_id,
        score=1.0,
        text="alpha failure",
        metadata=meta,
    )

    class DummyRetriever:
        def __init__(self, hits):
            self._hits = hits

        def retrieve(self, query, top_k=5, dbs=None):  # pragma: no cover - simple stub
            return self._hits, 1.0, None

    retr = Retriever(retriever=DummyRetriever([hit]))

    results = retr.search("alpha")
    assert results and results[0]["origin_db"] == "discrepancy"

    cb = ContextBuilder(retriever=retr)
    _ctx, _, vectors, meta_out = cb.build_context(
        "alpha", include_vectors=True, return_metadata=True
    )
    assert vectors and vectors[0][0] == "discrepancy" and vectors[0][1] == str(rec_id)
    assert meta_out["discrepancies"][0]["desc"] == "alpha failure"
