import sys
import types
import importlib.util
import pathlib
import pytest

from vector_service.retriever import PatchRetriever

_spec_vs = importlib.util.spec_from_file_location(
    "vector_service.vector_store", pathlib.Path("vector_service/vector_store.py")  # path-ignore
)
_vs_mod = importlib.util.module_from_spec(_spec_vs)
sys.modules["vector_service.vector_store"] = _vs_mod
_spec_vs.loader.exec_module(_vs_mod)
create_vector_store = _vs_mod.create_vector_store


class DummyVS:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def vectorise(self, kind, record):
        return [1.0, 0.0, 0.0]


@pytest.mark.parametrize(
    "backend,mod_name",
    [
        ("annoy", None),
        ("qdrant", "qdrant_client"),
        ("chroma", "chromadb"),
    ],
)
def test_patch_retriever_search_backends(monkeypatch, tmp_path, backend, mod_name):
    if mod_name:
        pytest.importorskip(mod_name)
    store_path = tmp_path / backend
    store = create_vector_store(3, store_path, backend=backend)
    if hasattr(store, "_save"):
        orig = store._save
        monkeypatch.setattr(store, "_save", lambda: None)
    store.add("patch", "1", [1.0, 0.0, 0.0], origin_db="patch", metadata={"diff": "example"})
    if hasattr(store, "_save"):
        store._save = orig
        store._save()
    vec_service = DummyVS(store)
    pr = PatchRetriever(store=store, vector_service=vec_service)
    res = pr.search("anything", top_k=1)
    assert res and res[0]["record_id"] == "1"
    assert 0.0 <= res[0]["score"] <= 1.0


def test_reload_from_config(monkeypatch, tmp_path):
    store = create_vector_store(3, tmp_path / "idx.ann", backend="annoy")
    vec_service = DummyVS(store)

    cfg = types.SimpleNamespace(
        vector_store=types.SimpleNamespace(
            backend="annoy", path=str(tmp_path / "idx.ann"), metric="cosine"
        ),
        vector=types.SimpleNamespace(dimensions=3),
    )
    module = types.SimpleNamespace(CONFIG=cfg)
    monkeypatch.setitem(sys.modules, "config", module)

    pr = PatchRetriever(store=store, vector_service=vec_service)
    assert pr.metric == "cosine"
    cfg.vector_store.metric = "inner_product"
    pr.reload_from_config()
    assert pr.metric == "inner_product"
