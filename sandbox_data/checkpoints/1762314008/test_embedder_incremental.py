import sys
import types
from pathlib import Path

from db_router import DBRouter

# Provide minimal PatchHistoryDB stub for embedder
code_db_mod = types.ModuleType("code_database")


class PatchHistoryDB:  # noqa: D401 - simple stub
    """SQLite-backed patch history for tests."""

    def __init__(self, path):
        self.path = Path(path)
        self.router = DBRouter("patch_history", str(self.path), str(self.path))


code_db_mod.PatchHistoryDB = PatchHistoryDB
sys.modules.setdefault("code_database", code_db_mod)

from vector_service.embedder import Embedder  # noqa: E402


class DummyStore:
    def __init__(self):
        self.meta = []

    def add(self, kind, record_id, vector, *, origin_db=None, metadata=None):
        self.meta.append(
            {"type": kind, "id": record_id, "origin_db": origin_db, "metadata": dict(metadata or {})}
        )


class DummyService:
    def __init__(self, store):
        self.vector_store = store

    def vectorise_and_store(self, kind, record_id, record, *, origin_db=None, metadata=None):
        self.vector_store.add(kind, record_id, [0.0], origin_db=origin_db, metadata=metadata)


def _setup_db(tmp_path):
    db_path = tmp_path / "patch_history.db"
    phdb = PatchHistoryDB(db_path)
    conn = phdb.router.get_connection("patch_history")
    conn.execute(
        "CREATE TABLE patch_history (id INTEGER PRIMARY KEY AUTOINCREMENT, description TEXT, diff TEXT, summary TEXT, timestamp REAL, enhancement_name TEXT)"
    )
    return db_path, conn


def test_rerun_skips_existing(tmp_path):
    db_path, conn = _setup_db(tmp_path)
    conn.execute(
        "INSERT INTO patch_history (description, diff, summary, timestamp, enhancement_name) VALUES (?,?,?,?,?)",
        ("d1", "df1", "s1", 0.0, "e1"),
    )
    conn.execute(
        "INSERT INTO patch_history (description, diff, summary, timestamp, enhancement_name) VALUES (?,?,?,?,?)",
        ("d2", "df2", "s2", 0.0, "e2"),
    )
    conn.commit()
    store = DummyStore()
    svc = DummyService(store)
    emb = Embedder(db_path, svc=svc)
    emb.embed_all()
    assert [m["metadata"]["patch_id"] for m in store.meta] == [1, 2]
    conn.execute(
        "INSERT INTO patch_history (description, diff, summary, timestamp, enhancement_name) VALUES (?,?,?,?,?)",
        ("d3", "df3", "s3", 0.0, "e3"),
    )
    conn.commit()
    emb.embed_all()
    assert [m["metadata"]["patch_id"] for m in store.meta] == [1, 2, 3]


def test_errors_do_not_abort(tmp_path, caplog):
    db_path, conn = _setup_db(tmp_path)
    for i in range(3):
        conn.execute(
            "INSERT INTO patch_history (description, diff, summary, timestamp, enhancement_name) VALUES (?,?,?,?,?)",
            (f"d{i}", f"df{i}", f"s{i}", 0.0, f"e{i}"),
        )
    conn.commit()

    store = DummyStore()

    class FlakyService(DummyService):
        def vectorise_and_store(self, kind, record_id, record, *, origin_db=None, metadata=None):
            if record_id == "2":
                raise RuntimeError("boom")
            super().vectorise_and_store(kind, record_id, record, origin_db=origin_db, metadata=metadata)

    svc = FlakyService(store)
    emb = Embedder(db_path, svc=svc)
    with caplog.at_level("WARNING"):
        emb.embed_all()
    ids = [m["metadata"]["patch_id"] for m in store.meta]
    assert ids == [1, 3]
    assert any("failed to embed patch 2" in r.message for r in caplog.records)
