import logging
import pytest
from menace import MenaceDB
from db_dedup import insert_if_unique

sa = pytest.importorskip("sqlalchemy")


@pytest.mark.parametrize(
    "table,values,hash_fields",
    [
        (
            "bots",
            {
                "bot_name": "alpha",
                "bot_type": "worker",
                "assigned_task": "[]",
                "dependencies": "[]",
                "resource_estimates": "{}",
            },
            [
                "bot_name",
                "bot_type",
                "assigned_task",
                "dependencies",
                "resource_estimates",
            ],
        ),
        (
            "workflows",
            {
                "workflow_name": "wf1",
                "task_tree": "[]",
                "dependencies": "[]",
                "resource_allocation_plan": "{}",
                "status": "active",
            },
            [
                "workflow_name",
                "task_tree",
                "dependencies",
                "resource_allocation_plan",
                "status",
            ],
        ),
        (
            "enhancements",
            {
                "description_of_change": "desc",
                "reason_for_change": "why",
                "performance_delta": 0.1,
                "timestamp": "now",
                "triggered_by": "unit",
                "source_menace_id": "m1",
            },
            [
                "description_of_change",
                "reason_for_change",
                "performance_delta",
                "timestamp",
                "triggered_by",
            ],
        ),
        (
            "errors",
            {
                "error_type": "t1",
                "error_description": "msg1",
                "resolution_status": "open",
            },
            ["error_type", "error_description", "resolution_status"],
        ),
    ],
)
def test_menacedb_dedup(tmp_path, caplog, table, values, hash_fields):
    mdb = MenaceDB(url=f"sqlite:///{tmp_path / 'menace.db'}")
    logger = logging.getLogger(__name__)
    tbl = getattr(mdb, table)

    id1 = insert_if_unique(
        tbl,
        values,
        hash_fields,
        "m1",
        engine=mdb.engine,
        logger=logger,
    )
    assert id1 is not None

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        id2 = insert_if_unique(
            tbl,
            values,
            hash_fields,
            "m1",
            engine=mdb.engine,
            logger=logger,
        )
    assert id2 == id1

    with mdb.engine.begin() as conn:
        count = conn.execute(sa.select(sa.func.count()).select_from(tbl)).scalar()
    assert count == 1
    assert any(
        f"Duplicate insert ignored for {table}" in r.message for r in caplog.records
    )
