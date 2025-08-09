import types
from menace.mutation_lineage import MutationLineage
from menace.evolution_history_db import EvolutionHistoryDB, EvolutionEvent
from menace.code_database import PatchHistoryDB, PatchRecord


def test_tree_backtrack_and_clone(tmp_path):
    e_db = EvolutionHistoryDB(tmp_path / "e.db")
    p_db = PatchHistoryDB(tmp_path / "p.db")

    root_patch = PatchRecord(filename="a.py", description="root", roi_before=0, roi_after=1)
    root_id = p_db.add(root_patch)
    root_event = EvolutionEvent("root", 0, 1, 1.0, patch_id=root_id, workflow_id=1)
    root_event_id = e_db.add(root_event)

    bad_patch = PatchRecord(
        filename="a.py",
        description="bad",
        roi_before=1,
        roi_after=0.5,
        roi_delta=-0.5,
        parent_patch_id=root_id,
    )
    bad_id = p_db.add(bad_patch)
    bad_event = EvolutionEvent(
        "bad",
        1,
        0.5,
        -0.5,
        patch_id=bad_id,
        workflow_id=1,
        parent_event_id=root_event_id,
    )
    e_db.add(bad_event)

    ml = MutationLineage(history_db=e_db, patch_db=p_db)
    tree = ml.build_tree(1)
    assert tree and tree[0]["patch_id"] == root_id
    assert tree[0]["children"][0]["patch_id"] == bad_id

    path = ml.backtrack_failed_path(bad_id)
    assert path == [bad_id, root_id]

    clone_id = ml.clone_branch_for_ab_test(root_id, "variant")
    with p_db._connect() as conn:  # type: ignore[attr-defined]
        parent = conn.execute("SELECT parent_patch_id FROM patch_history WHERE id=?", (clone_id,)).fetchone()[0]
    assert parent == root_id
