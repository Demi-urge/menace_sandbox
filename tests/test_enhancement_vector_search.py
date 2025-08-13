import menace.chatgpt_enhancement_bot as ceb
import pytest


@pytest.mark.parametrize("backend", ["annoy", "faiss"])
def test_vector_search(tmp_path, backend):
    if backend == "faiss":
        pytest.importorskip("faiss")
    db = ceb.EnhancementDB(
        tmp_path / "e.db",
        vector_backend=backend,
        vector_index_path=tmp_path / f"idx.{backend}.index",
    )
    db._embed = lambda text: ([float(len(text)), 0.0] if len(text) % 2 == 0 else [0.0, float(len(text))])
    e1 = ceb.Enhancement(idea="i1", rationale="r1", summary="alpha", before_code="a", after_code="b")
    id1 = db.add(e1)
    e2 = ceb.Enhancement(idea="i2", rationale="r2", summary="beta", before_code="c", after_code="d")
    id2 = db.add(e2)

    vec1 = db.vector(id1)
    assert vec1 is not None
    results = db.search_by_vector(vec1, top_k=2)
    assert results and results[0].summary == "alpha"

    # ensure metadata stored
    row = db.conn.execute(
        "SELECT kind FROM enhancement_embeddings WHERE record_id=?", (id1,),
    ).fetchone()
    assert row and row[0] == "enhancement"

    # update second enhancement and ensure embedding refreshed
    e2.summary = "completely different"
    db.update(id2, e2)
    vec2 = db.vector(id2)
    results2 = db.search_by_vector(vec2, top_k=1)
    assert results2 and results2[0].summary == "completely different"
