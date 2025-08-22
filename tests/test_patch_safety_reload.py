from patch_safety import PatchSafety
from vector_service.context_builder import ContextBuilder
from vector_service.patch_logger import PatchLogger
from vector_service.cognition_layer import CognitionLayer


class DummyRetriever:
    def search(self, *args, **kwargs):
        return []


def test_failures_refresh_evaluation(tmp_path):
    store = tmp_path / "failures.jsonl"
    ps_logger = PatchSafety(storage_path=str(store), failure_db_path=None, refresh_interval=0)
    ps_builder = PatchSafety(storage_path=str(store), failure_db_path=None, refresh_interval=0)
    patch_logger = PatchLogger(patch_safety=ps_logger)
    builder = ContextBuilder(retriever=DummyRetriever(), patch_safety=ps_builder)
    layer = CognitionLayer(
        retriever=DummyRetriever(),
        context_builder=builder,
        patch_logger=patch_logger,
        vector_metrics=None,
        roi_tracker=None,
    )
    err_meta = {"category": "fail"}
    sid = "s1"
    layer._session_vectors[sid] = [("error", "1", 0.1)]
    layer._retrieval_meta[sid] = {"error:1": err_meta}
    ok1, _, _ = ps_builder.evaluate({}, err_meta, origin="error")
    assert ok1
    layer.record_patch_outcome(sid, False)
    ok2, score2, _ = ps_builder.evaluate({}, err_meta, origin="error")
    assert not ok2
    assert score2 >= ps_builder.threshold
