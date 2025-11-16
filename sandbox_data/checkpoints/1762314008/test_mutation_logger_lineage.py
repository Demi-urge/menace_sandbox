import pytest

from menace import mutation_logger as ml
from menace.evolution_history_db import EvolutionHistoryDB


def _setup_db(tmp_path, monkeypatch):
    db = EvolutionHistoryDB(tmp_path / "e.db")
    monkeypatch.setattr(ml, "_history_db", db)
    monkeypatch.setattr(ml, "_event_bus", None)
    return db


def test_log_mutation_records_metrics_and_parent(tmp_path, monkeypatch):
    db = _setup_db(tmp_path, monkeypatch)
    root_id = ml.log_mutation(
        "root", "initial", "manual", 1.0, workflow_id=1, before_metric=1.0, after_metric=2.0
    )
    child_id = ml.log_mutation(
        "child",
        "refactor",
        "auto",
        0.5,
        workflow_id=1,
        parent_id=root_id,
        before_metric=2.0,
        after_metric=3.0,
    )

    row_root = db.conn.execute(
        'SELECT reason, "trigger", parent_event_id, before_metric, after_metric, roi FROM evolution_history WHERE rowid=?',
        (root_id,),
    ).fetchone()
    row_child = db.conn.execute(
        'SELECT reason, "trigger", parent_event_id, before_metric, after_metric, roi FROM evolution_history WHERE rowid=?',
        (child_id,),
    ).fetchone()

    assert row_root == ("initial", "manual", None, 1.0, 2.0, 1.0)
    assert row_child == ("refactor", "auto", root_id, 2.0, 3.0, 1.0)


def test_build_lineage_reconstructs_tree(tmp_path, monkeypatch):
    _setup_db(tmp_path, monkeypatch)
    root = ml.log_mutation(
        "root",
        "r",
        "t",
        1.0,
        workflow_id=1,
        before_metric=0.0,
        after_metric=1.0,
    )
    child = ml.log_mutation(
        "child",
        "r",
        "t",
        1.0,
        workflow_id=1,
        parent_id=root,
        before_metric=1.0,
        after_metric=2.0,
    )
    grand = ml.log_mutation(
        "grand",
        "r",
        "t",
        1.0,
        workflow_id=1,
        parent_id=child,
        before_metric=2.0,
        after_metric=3.0,
    )

    tree = ml.build_lineage(1)
    assert tree and tree[0]["rowid"] == root
    assert tree[0]["children"][0]["rowid"] == child
    assert tree[0]["children"][0]["children"][0]["rowid"] == grand
