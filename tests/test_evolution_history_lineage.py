from menace.evolution_history_db import EvolutionHistoryDB, EvolutionEvent


def test_subtree_and_spawn(tmp_path):
    db = EvolutionHistoryDB(tmp_path / "e.db")
    root_id = db.add(EvolutionEvent("root", 0, 1, 1.0, workflow_id=1))
    child_id = db.spawn_variant(root_id, "child")
    tree = db.subtree(root_id)
    assert tree and tree["rowid"] == root_id
    assert tree["children"][0]["rowid"] == child_id
    row = db.conn.execute(
        "SELECT parent_event_id, workflow_id FROM evolution_history WHERE rowid=?",
        (child_id,),
    ).fetchone()
    assert row == (root_id, 1)
