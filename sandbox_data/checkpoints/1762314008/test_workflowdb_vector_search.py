from types import MethodType

import pytest

from menace.task_handoff_bot import WorkflowDB, WorkflowRecord


@pytest.mark.parametrize("backend", ["annoy", "faiss"])
def test_embedding_workflow(tmp_path, backend):
    if backend == "faiss":
        pytest.importorskip("faiss")
        pytest.importorskip("numpy")
    db = WorkflowDB(
        path=tmp_path / "wf.db",
        vector_backend=backend,
        vector_index_path=tmp_path / f"wf.{backend}.index",
    )
    db.metadata_path = tmp_path / "wf.meta.json"
    db._metadata.clear()
    db._id_map.clear()
    db._index = None

    captured: list[str] = []

    def fake_embed(self, text: str):
        captured.append(text)
        if "alpha" in text:
            return [1.0, 0.0]
        if "beta" in text:
            return [0.0, 1.0]
        return [1.0, 1.0]

    db._embed = MethodType(fake_embed, db)

    rec1 = WorkflowRecord(workflow=["alpha"], task_sequence=["step1"], title="Alpha")

    id1 = db.add(rec1)

    assert str(id1) in db._metadata
    assert db._metadata[str(id1)]["source_id"] == str(id1)
    assert len(captured) == 1

    res1 = db.search_by_vector([1.0, 0.0], top_k=1)
    assert res1 and res1[0].title == "Alpha"

    # insert record without embedding and backfill
    db.conn.execute(
        "INSERT INTO workflows(workflow, task_sequence, title, timestamp) VALUES (?,?,?,?)",
        ("beta", "s", "Beta", "0"),
    )
    db.conn.commit()
    new_id = db.conn.execute("SELECT id FROM workflows WHERE title='Beta'").fetchone()[0]
    assert str(new_id) not in db._metadata
    db.backfill_embeddings()
    assert str(new_id) in db._metadata
    res2 = db.search_by_vector([0.0, 1.0], top_k=1)
    assert res2 and res2[0].title == "Beta"
