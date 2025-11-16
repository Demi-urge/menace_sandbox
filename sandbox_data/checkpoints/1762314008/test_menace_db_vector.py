import sqlite3
import pytest

from vector_service import EmbeddableDBMixin


class _MemoryDB(EmbeddableDBMixin):
    def __init__(self, path, backend, index_path):
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        EmbeddableDBMixin.__init__(
            self,
            backend=backend,
            index_path=index_path,
            metadata_path=index_path.with_suffix(".json"),
        )

    def vector(self, record):
        return list(record)


@pytest.mark.parametrize("backend", ["annoy", "faiss"])
def test_search_by_vector(tmp_path, backend):
    if backend == "faiss":
        pytest.importorskip("faiss")
    db = _MemoryDB(tmp_path / "mem.db", backend, tmp_path / f"mem.{backend}.index")
    db.add_embedding("a", [1.0, 0.0], "test", source_id="a")
    db.add_embedding("b", [0.0, 1.0], "test", source_id="b")
    assert db._metadata["a"]["source_id"] == "a"
    assert db._metadata["b"]["source_id"] == "b"
    r1 = db.search_by_vector([1.0, 0.0], top_k=1)
    assert r1 and r1[0][0] == "a"
    r2 = db.search_by_vector([0.0, 1.0], top_k=1)
    assert r2 and r2[0][0] == "b"

