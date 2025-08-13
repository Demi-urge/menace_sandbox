from types import MethodType

from menace.task_handoff_bot import WorkflowDB, WorkflowRecord


def test_workflow_vector_search(tmp_path):
    db = WorkflowDB(
        path=tmp_path / "wf.db",
        vector_backend="annoy",
        vector_index_path=tmp_path / "wf.index",
    )

    captured: list[str] = []

    def fake_embed(self, text: str):
        captured.append(text)
        if "alpha" in text:
            return [1.0, 0.0]
        return [0.0, 1.0]

    db._embed = MethodType(fake_embed, db)

    rec1 = WorkflowRecord(workflow=["alpha"], task_sequence=["step1"], title="Alpha")
    rec2 = WorkflowRecord(workflow=["beta"], task_sequence=["step2"], title="Beta")

    id1 = db.add(rec1)
    id2 = db.add(rec2)

    assert len(captured) == 2

    res1 = db.search_by_vector([1.0, 0.0], top_k=1)
    assert res1 and res1[0].title == "Alpha"
    res2 = db.search_by_vector([0.0, 1.0], top_k=1)
    assert res2 and res2[0].title == "Beta"

    db.update_status(id1, "done")
    db.update_statuses([id2], "done")
    assert len(captured) == 4
