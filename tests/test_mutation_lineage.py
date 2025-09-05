import pytest
from menace.mutation_lineage import MutationLineage
from menace.evolution_history_db import EvolutionHistoryDB, EvolutionEvent
from menace.code_database import PatchHistoryDB, PatchRecord


def test_tree_backtrack_and_clone(tmp_path):
    e_db = EvolutionHistoryDB(tmp_path / "e.db")
    p_db = PatchHistoryDB(tmp_path / "p.db")

    root_patch = PatchRecord(filename="a.py", description="root", roi_before=0, roi_after=1)  # path-ignore
    root_id = p_db.add(root_patch)
    root_event = EvolutionEvent("root", 0, 1, 1.0, patch_id=root_id, workflow_id=1)
    root_event_id = e_db.add(root_event)

    bad_patch = PatchRecord(
        filename="a.py",  # path-ignore
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
        query = "SELECT parent_patch_id FROM patch_history WHERE id=?"
        parent = conn.execute(query, (clone_id,)).fetchone()[0]
    assert parent == root_id


def test_backtrack_failed_path_multilevel(tmp_path):
    e_db = EvolutionHistoryDB(tmp_path / "e.db")
    p_db = PatchHistoryDB(tmp_path / "p.db")

    root = PatchRecord(
        filename="a.py",  # path-ignore
        description="root",
        roi_before=0,
        roi_after=1,
        roi_delta=1,
    )
    root_id = p_db.add(root)
    bad1 = PatchRecord(
        filename="a.py",  # path-ignore
        description="bad1",
        roi_before=1,
        roi_after=0.5,
        roi_delta=-0.5,
        parent_patch_id=root_id,
    )
    bad1_id = p_db.add(bad1)
    bad2 = PatchRecord(
        filename="a.py",  # path-ignore
        description="bad2",
        roi_before=0.5,
        roi_after=0.2,
        roi_delta=-0.3,
        parent_patch_id=bad1_id,
    )
    bad2_id = p_db.add(bad2)

    ml = MutationLineage(history_db=e_db, patch_db=p_db)
    path = ml.backtrack_failed_path(bad2_id)
    assert path == [bad2_id, bad1_id, root_id]


def test_clone_branch_for_ab_test_copies_fields(tmp_path):
    e_db = EvolutionHistoryDB(tmp_path / "e.db")
    p_db = PatchHistoryDB(tmp_path / "p.db")

    original = PatchRecord(
        filename="a.py",  # path-ignore
        description="orig",
        roi_before=1.0,
        roi_after=1.5,
        errors_before=3,
        errors_after=1,
        complexity_before=2.0,
        complexity_after=2.5,
        roi_delta=0.5,
        trending_topic="topic",
        code_id=7,
        code_hash="hash",
        source_bot="bot",
        version="v1",
    )
    root_id = p_db.add(original)
    ml = MutationLineage(history_db=e_db, patch_db=p_db)

    clone_id = ml.clone_branch_for_ab_test(root_id, "variant")
    with p_db._connect() as conn:  # type: ignore[attr-defined]
        row = conn.execute(
            (
                "SELECT filename, description, roi_before, roi_after, errors_before, errors_after, "
                "complexity_before, complexity_after, trending_topic, code_id, code_hash, "
                "source_bot, version, parent_patch_id FROM patch_history WHERE id=?"
            ),
            (clone_id,),
        ).fetchone()

    assert row is not None
    (
        filename,
        description,
        roi_before,
        roi_after,
        errors_before,
        errors_after,
        complexity_before,
        complexity_after,
        trending_topic,
        code_id,
        code_hash,
        source_bot,
        version,
        parent_patch_id,
    ) = row

    assert filename == original.filename
    assert description == "variant"
    assert roi_before == pytest.approx(original.roi_after)
    assert roi_after == pytest.approx(original.roi_after)
    assert errors_before == original.errors_after
    assert errors_after == original.errors_after
    assert complexity_before == pytest.approx(original.complexity_after)
    assert complexity_after == pytest.approx(original.complexity_after)
    assert trending_topic == original.trending_topic
    assert code_id == original.code_id
    assert code_hash == original.code_hash
    assert source_bot == original.source_bot
    assert version == original.version
    assert parent_patch_id == root_id


def test_render_tree_generates_dot(tmp_path):
    e_db = EvolutionHistoryDB(tmp_path / "e.db")
    p_db = PatchHistoryDB(tmp_path / "p.db")

    patch = PatchRecord(filename="a.py", description="root", roi_before=0, roi_after=1)  # path-ignore
    pid = p_db.add(patch)
    event = EvolutionEvent("root", 0, 1, 1.0, patch_id=pid, workflow_id=1)
    e_db.add(event)

    ml = MutationLineage(history_db=e_db, patch_db=p_db)
    out = tmp_path / "tree.dot"
    ml.render_tree(1, out)
    assert out.exists()
    assert "digraph" in out.read_text()
