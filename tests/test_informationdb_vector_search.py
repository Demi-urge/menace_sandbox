from types import MethodType

import pytest

from menace.information_db import InformationDB, InformationRecord


@pytest.mark.parametrize("backend", ["annoy", "faiss"])
def test_embedding_workflow(tmp_path, backend):
    if backend == "faiss":
        pytest.importorskip("faiss")
        pytest.importorskip("numpy")
    db = InformationDB(
        path=str(tmp_path / "info.db"),
        vector_backend=backend,
        vector_index_path=str(tmp_path / f"info.{backend}.index"),
    )

    def fake_embed(self, text: str):
        if "alpha" in text:
            return [1.0, 0.0]
        if "beta" in text:
            return [0.0, 1.0]
        return [1.0, 1.0]

    db._embed = MethodType(fake_embed, db)

    rec = InformationRecord(data_type="news", summary="alpha", keywords=["k1"])
    info_id = db.add(rec)
    assert str(info_id) in db._metadata
    res1 = db.search_by_vector([1.0, 0.0], top_k=1)
    assert res1 and res1[0]["summary"] == "alpha"

    db.conn.execute(
        "INSERT INTO information(data_type, summary, keywords, source_menace_id) VALUES (?,?,?,?)",
        ("news", "beta", "", db.router.menace_id),
    )
    db.conn.commit()
    new_id = db.conn.execute("SELECT info_id FROM information WHERE summary='beta'").fetchone()[0]
    assert str(new_id) not in db._metadata
    db.backfill_embeddings()
    assert str(new_id) in db._metadata
    res2 = db.search_by_vector([0.0, 1.0], top_k=1)
    assert res2 and res2[0]["summary"] == "beta"
