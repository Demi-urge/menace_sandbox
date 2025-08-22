from types import MethodType

import pytest

from patch_suggestion_db import PatchSuggestionDB, SuggestionRecord


@pytest.mark.parametrize("backend", ["annoy", "faiss"])
def test_patch_suggestion_embedding(tmp_path, backend):
    if backend == "faiss":
        pytest.importorskip("faiss")
        pytest.importorskip("numpy")
    db = PatchSuggestionDB(
        path=tmp_path / "s.db",
        vector_backend=backend,
        vector_index_path=tmp_path / f"s.{backend}.index",
    )

    captured: list[str] = []

    def fake_encode(self, text: str):
        captured.append(text)
        if "alpha" in text:
            return [1.0, 0.0]
        if "beta" in text:
            return [0.0, 1.0]
        return [1.0, 1.0]

    db.encode_text = MethodType(fake_encode, db)

    rec1 = SuggestionRecord(module="m1", description="alpha fix")
    id1 = db.add(rec1)
    assert str(id1) in db._metadata
    assert captured and "alpha" in captured[0]

    res1 = db.search_by_vector([1.0, 0.0], top_k=1)
    assert res1 and int(res1[0][0]) == id1

    db.conn.execute(
        "INSERT INTO suggestions(module, description, ts) VALUES (?,?,?)",
        ("m2", "beta change", "t"),
    )
    db.conn.commit()
    id2 = db.conn.execute("SELECT id FROM suggestions WHERE module='m2'").fetchone()[0]
    assert str(id2) not in db._metadata
    db.backfill_embeddings()
    assert str(id2) in db._metadata
    res2 = db.search_by_vector([0.0, 1.0], top_k=1)
    assert res2 and int(res2[0][0]) == id2

