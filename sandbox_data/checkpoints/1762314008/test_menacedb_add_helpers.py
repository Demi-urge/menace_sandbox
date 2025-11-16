import logging
import pytest
from menace import MenaceDB

sa = pytest.importorskip("sqlalchemy")


@pytest.mark.parametrize(
    "method,table,kwargs",
    [
        (
            "add_bot",
            "bots",
            {
                "bot_name": "alpha",
                "bot_type": "worker",
                "assigned_task": "[]",
                "dependencies": "[]",
                "resource_estimates": "{}",
            },
        ),
        (
            "add_workflow",
            "workflows",
            {
                "workflow_name": "wf1",
                "task_tree": "[]",
                "dependencies": "[]",
                "resource_allocation_plan": "{}",
                "status": "active",
            },
        ),
        (
            "add_enhancement",
            "enhancements",
            {
                "description_of_change": "desc",
                "reason_for_change": "why",
                "performance_delta": 0.1,
                "timestamp": "now",
                "triggered_by": "unit",
                "source_menace_id": "m1",
            },
        ),
    ],
)

def test_menacedb_add_helpers_dedup(tmp_path, caplog, method, table, kwargs):
    db = MenaceDB(url=f"sqlite:///{tmp_path / 'menace.db'}")
    func = getattr(db, method)
    id1 = func(**kwargs)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        id2 = func(**kwargs)
    assert id1 == id2
    tbl = getattr(db, table)
    with db.engine.begin() as conn:
        count = conn.execute(sa.select(sa.func.count()).select_from(tbl)).scalar()
    assert count == 1
    assert any(
        f"Duplicate insert ignored for {table}" in r.message for r in caplog.records
    )
