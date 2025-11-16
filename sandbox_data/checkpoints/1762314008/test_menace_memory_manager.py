import menace.menace_memory_manager as mmm
import logging


def test_versioning(tmp_path):
    manager = mmm.MenaceMemoryManager(tmp_path / "m.db")
    v1 = manager.next_version("k1")
    manager.log(mmm.MemoryEntry("k1", "d1", v1, "t"))
    v2 = manager.next_version("k1")
    manager.log(mmm.MemoryEntry("k1", "d2", v2, "t"))
    rows = manager.query("k1", limit=2)
    assert [r.data for r in rows] == ["d2", "d1"]


def test_subscriber_called(tmp_path):
    manager = mmm.MenaceMemoryManager(tmp_path / "m.db")
    called = {}

    def cb(entry: mmm.MemoryEntry) -> None:
        called["hit"] = entry.data

    manager.subscribe(cb)
    manager.log(mmm.MemoryEntry("k", "d", 1, "t"))
    assert called.get("hit") == "d"


def test_query_vector_fallback(tmp_path):
    manager = mmm.MenaceMemoryManager(tmp_path / "m.db", embedder=None)
    manager.store("k1", "hello world")
    manager.store("k2", "another entry")
    res = manager.query_vector("hello")
    assert res


def test_faiss_backend_without_lib(tmp_path, monkeypatch):
    monkeypatch.setattr(mmm, "faiss", None)
    manager = mmm.MenaceMemoryManager(
        tmp_path / "m.db", embedder=None, cluster_backend="faiss", recluster_interval=1
    )
    manager.store("k1", "d1")  # should fall back without error
    assert manager.query("k1")


def test_faiss_mock_clustering(tmp_path, monkeypatch):
    import types

    class DummyIndex:
        def __init__(self):
            self.ids = []
        def add_with_ids(self, vecs, ids):
            self.ids.extend(getattr(ids, "tolist", lambda: ids)())
        def search(self, x, k):
            return [[0.0] * k], [self.ids[:k] or [-1] * k]

    class DummyKMeans:
        def __init__(self, d, k, niter=20, verbose=False):
            self.index = DummyIndex()
        def train(self, x):
            pass

    dummy = types.SimpleNamespace(
        Kmeans=DummyKMeans,
        IndexFlatL2=lambda d: DummyIndex(),
        IndexIDMap=lambda idx: idx,
    )
    monkeypatch.setattr(mmm, "faiss", dummy)
    manager = mmm.MenaceMemoryManager(
        tmp_path / "m.db", embedder=None, cluster_backend="faiss", recluster_interval=1
    )
    manager._embed = lambda t: [0.0]
    manager.store("k1", "d1")
    assert manager._faiss_index is not None


def test_annoy_vector_backend(tmp_path):
    import annoy  # type: ignore
    manager = mmm.MenaceMemoryManager(
        tmp_path / "m.db",
        embedder=None,
        vector_backend="annoy",
        vector_index_path=tmp_path / "vec.ann",
    )
    manager._embed = lambda t: [1.0, 0.0] if "first" in t else [0.0, 1.0]
    manager.store("k1", "first entry")
    manager.store("k2", "second entry")
    manager.migrate_embeddings_to_index()
    res = manager.query_vector("first entry", limit=1)
    assert res and res[0].key == "k1"


def test_vector_backend_logs_failure(tmp_path, caplog, monkeypatch):
    manager = mmm.MenaceMemoryManager(
        tmp_path / "m.db",
        embedder=None,
        vector_backend="annoy",
        vector_index_path=tmp_path / "vec.ann",
    )
    manager._embed = lambda t: [0.1, 0.2]
    manager._index_add = lambda *a, **k: None
    manager.store("k1", "text")

    class FailIndex:
        def get_nns_by_vector(self, *a, **k):
            raise RuntimeError("boom")

    manager._vector_index = FailIndex()
    caplog.set_level(logging.ERROR)
    res = manager.query_vector("text")
    assert res
    assert "annoy vector search failed" in caplog.text
