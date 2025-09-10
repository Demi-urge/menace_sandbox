import sys
import types

import pytest

dummy_patch_logger = types.ModuleType("vector_service.patch_logger")

dummy_patch_logger._VECTOR_RISK = 0
sys.modules.setdefault("vector_service.patch_logger", dummy_patch_logger)


class DummyRetriever:
    def search(self, *args, **kwargs):
        return []


def make_builder(monkeypatch):
    import vector_service.context_builder as cb

    monkeypatch.setattr(cb.ContextBuilder, "refresh_db_weights", lambda self, *a, **k: None)
    monkeypatch.setattr(cb, "_ensure_vector_service", lambda: None)
    dummy_patch = types.SimpleNamespace(search=lambda *a, **k: [])
    builder = cb.ContextBuilder(retriever=DummyRetriever(), patch_retriever=dummy_patch, db_weights={})
    monkeypatch.setattr(builder.patch_safety, "load_failures", lambda: None)
    return builder, cb


def test_builder_falls_back_to_core_dbs(monkeypatch):
    builder, cb = make_builder(monkeypatch)
    called: dict[str, list[str]] = {}

    def fake_ensure(dbs):
        called["dbs"] = list(dbs)

    monkeypatch.setattr(cb, "ensure_embeddings_fresh", fake_ensure)
    builder.build("query")
    assert set(called["dbs"]) == {"code", "bot", "error", "workflow"}


def test_builder_surfaces_stale_embedding_error(monkeypatch):
    builder, cb = make_builder(monkeypatch)

    def fake_ensure(dbs):
        raise cb.StaleEmbeddingsError({"code": "missing"})

    monkeypatch.setattr(cb, "ensure_embeddings_fresh", fake_ensure)
    with pytest.raises(RuntimeError) as exc:
        builder.build("query")
    assert "code" in str(exc.value)
