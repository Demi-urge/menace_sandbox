import menace.chatgpt_enhancement_bot as ceb
import pytest


@pytest.mark.parametrize("backend", ["annoy", "faiss"])
def test_embedding_workflow(tmp_path, backend):
    if backend == "faiss":
        pytest.importorskip("faiss")
        pytest.importorskip("numpy")
    db = ceb.EnhancementDB(
        tmp_path / "e.db",
        vector_backend=backend,
        vector_index_path=tmp_path / f"idx.{backend}.index",
    )

    def fake_embed(text: str):
        if "alpha" in text:
            return [1.0, 0.0]
        if "beta" in text:
            return [0.0, 1.0]
        return [1.0, 1.0]

    db._embed = fake_embed

    e1 = ceb.Enhancement(idea="i1", rationale="r1", summary="alpha", before_code="a", after_code="b")
    id1 = db.add(e1)

    assert str(id1) in db._metadata
    res1 = db.search_by_vector([1.0, 0.0], top_k=1)
    assert res1 and res1[0].summary == "alpha"

    # insert enhancement without embedding and backfill
    db.conn.execute(
        "INSERT INTO enhancements(idea, rationale, summary, before_code, after_code, source_menace_id) VALUES (?,?,?,?,?,?)",
        ("i2", "r2", "beta", "x", "y", db.router.menace_id),
    )
    db.conn.commit()
    new_id = db.conn.execute(
        "SELECT id FROM enhancements WHERE summary='beta' AND source_menace_id=?",
        (db.router.menace_id,),
    ).fetchone()[0]
    assert str(new_id) not in db._metadata
    db.backfill_embeddings()
    assert str(new_id) in db._metadata
    res2 = db.search_by_vector([0.0, 1.0], top_k=1)
    assert res2 and res2[0].summary == "beta"

    # ensure metadata mirrored in SQLite
    row = db.conn.execute(
        "SELECT kind FROM enhancement_embeddings WHERE record_id=?", (id1,),
    ).fetchone()
    assert row and row[0] == "enhancement"
    row2 = db.conn.execute(
        "SELECT kind FROM enhancement_embeddings WHERE record_id=?", (new_id,),
    ).fetchone()
    assert row2 and row2[0] == "enhancement"
