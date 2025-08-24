from __future__ import annotations

from db_router import DBRouter
from workflow_summary_db import WorkflowSummaryDB


def test_workflow_summary_scope_utils_filters(tmp_path):
    shared = tmp_path / "shared.db"
    router1 = DBRouter("one", str(tmp_path / "local1.db"), str(shared))
    db1 = WorkflowSummaryDB(router=router1)
    db1.set_summary(1, "alpha")

    router2 = DBRouter("two", str(tmp_path / "local2.db"), str(shared))
    db2 = WorkflowSummaryDB(router=router2)
    db2.set_summary(2, "beta")

    assert db1.get_summary(1) == "alpha"
    assert db2.get_summary(2) == "beta"
    assert db1.get_summary(2) is None
    assert db1.get_summary(2, scope="global") == "beta"

    assert [s.workflow_id for s in db1.all_summaries()] == [1]
    assert {s.workflow_id for s in db1.all_summaries(scope="global")} == {2}
    assert {s.workflow_id for s in db1.all_summaries(scope="all")} == {1, 2}
