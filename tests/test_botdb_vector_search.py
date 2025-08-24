from types import MethodType

import pytest

from menace.bot_database import BotDB, BotRecord
import db_router


@pytest.mark.parametrize("backend", ["annoy", "faiss"])
def test_embedding_workflow(tmp_path, backend):
    if backend == "faiss":
        pytest.importorskip("faiss")
        pytest.importorskip("numpy")
    db = BotDB(
        path=tmp_path / "b.db",
        vector_backend=backend,
        vector_index_path=tmp_path / f"b.{backend}.index",
    )
    db.metadata_path = tmp_path / "b.meta.json"
    db._metadata.clear()
    db._id_map.clear()
    db._index = None

    captured: list[str] = []

    def fake_encode(self, text: str):
        captured.append(text)
        if "alpha" in text:
            return [1.0, 0.0]
        if "beta" in text:
            return [0.0, 1.0]
        return [1.0, 1.0]

    db.encode_text = MethodType(fake_encode, db)

    rec1 = BotRecord(name="A", purpose="alpha", tags=["x"], toolchain=["tc1"])

    id1 = db.add_bot(rec1)
    assert str(id1) in db._metadata
    assert captured and "alpha" in captured[0] and "x" in captured[0] and "tc1" in captured[0]

    res1 = db.search_by_vector([1.0, 0.0], top_k=1)
    assert res1 and res1[0]["name"] == "A"

    # insert record without embedding and backfill
    db.conn.execute(
        "INSERT INTO bots(name, purpose, tags, toolchain, source_menace_id) VALUES (?,?,?,?,?)",
        ("B", "beta", "", "", db_router.GLOBAL_ROUTER.menace_id),
    )
    db.conn.commit()
    new_id = db.conn.execute("SELECT id FROM bots WHERE name='B'").fetchone()[0]
    assert str(new_id) not in db._metadata
    db.backfill_embeddings()
    assert str(new_id) in db._metadata
    res2 = db.search_by_vector([0.0, 1.0], top_k=1)
    assert res2 and res2[0]["name"] == "B"
