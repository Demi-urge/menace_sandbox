import json
import types

import vector_service.vectorizer as vz
from vector_service.retriever import PatchRetriever, Retriever
from vector_service.vector_store import AnnoyVectorStore
from vector_service.context_builder import ContextBuilder
from vector_service.cognition_layer import CognitionLayer
from patch_safety import PatchSafety


def test_nl_query_returns_patch_diff(monkeypatch, tmp_path):
    monkeypatch.setattr(vz, "load_handlers", lambda: {})

    def fake_vectorise(self, kind, record):
        mapping = {"fix bug": [1.0, 0.0, 0.0], "add feature": [0.0, 1.0, 0.0]}
        return mapping[record.get("text", "")]

    monkeypatch.setattr(vz.SharedVectorService, "vectorise", fake_vectorise, raising=False)

    store = AnnoyVectorStore(dim=3, path=tmp_path / "idx.ann")
    orig_save = store._save
    monkeypatch.setattr(store, "_save", lambda: None)
    store.add("patch", "1", [1.0, 0.0, 0.0], origin_db="patch", metadata={"diff": "fix bug diff"})
    store.add("patch", "2", [0.0, 1.0, 0.0], origin_db="patch", metadata={"diff": "add feature diff"})
    store._save = orig_save
    store._save()

    vec_service = vz.SharedVectorService(text_embedder=None, vector_store=store)
    patch_ret = PatchRetriever(store=store, vector_service=vec_service)

    retr = Retriever()
    monkeypatch.setattr(retr, "search", lambda *a, **k: [])

    cb = ContextBuilder(
        retriever=retr,
        patch_retriever=patch_ret,
        max_tokens=1000,
        patch_safety=PatchSafety(failure_db_path=None),
    )
    monkeypatch.setattr(cb, "refresh_db_weights", lambda *a, **k: None)
    monkeypatch.setattr(cb.patch_safety, "load_failures", lambda *a, **k: None)

    class DummyMetrics:
        def get_db_weights(self):
            return {}

        def log_retrieval(self, *a, **k):
            pass

        def save_session(self, *a, **k):
            pass

    layer = CognitionLayer(
        retriever=retr,
        patch_retriever=patch_ret,
        context_builder=cb,
        patch_logger=types.SimpleNamespace(roi_tracker=None, event_bus=None),
        vector_metrics=DummyMetrics(),
    )

    ctx, _sid = layer.query("fix bug", top_k=1)
    data = json.loads(ctx)
    assert "patches" in data
    assert any("fix bug" in p.get("desc", "") for p in data["patches"])

