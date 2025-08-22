from types import MethodType

import pytest

from contrarian_db import ContrarianDB, ContrarianRecord


@pytest.mark.parametrize("backend", ["annoy", "faiss"])
def test_contrarian_embedding_workflow(tmp_path, backend):
    if backend == "faiss":
        pytest.importorskip("faiss")
        pytest.importorskip("numpy")
    db = ContrarianDB(
        tmp_path / "c.db",
        vector_backend=backend,
        vector_index_path=tmp_path / f"c.{backend}.index",
    )

    captured: list[str] = []

    def fake_encode(self, text: str):
        captured.append(text)
        if "risk" in text:
            return [1.0, 0.0]
        if "reward" in text:
            return [0.0, 1.0]
        return [1.0, 1.0]

    db.encode_text = MethodType(fake_encode, db)

    rec1 = ContrarianRecord(innovation_name="risk", innovation_type="t")
    id1 = db.add(rec1)
    assert str(id1) in db._metadata
    assert captured and "risk" in captured[0]

    res1 = db.search_by_vector([1.0, 0.0], top_k=1)
    assert res1 and int(res1[0][0]) == id1

    db.conn.execute(
        "INSERT INTO contrarian_experiments(innovation_name, innovation_type, risk_score, reward_score, activation_trigger, resource_allocation, timestamp_created, timestamp_last_evaluated, status) VALUES (?,?,?,?,?,?,?,?,?)",
        ("reward", "t", 0.0, 0.0, "", "{}", "0", "", "active"),
    )
    db.conn.commit()
    id2 = db.conn.execute(
        "SELECT contrarian_id FROM contrarian_experiments WHERE innovation_name='reward'"
    ).fetchone()[0]
    assert str(id2) not in db._metadata
    db.backfill_embeddings()
    assert str(id2) in db._metadata
    res2 = db.search_by_vector([0.0, 1.0], top_k=1)
    assert res2 and int(res2[0][0]) == id2

