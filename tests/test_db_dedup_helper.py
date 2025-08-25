import importlib
import logging
import sys

import sqlalchemy as sa
if not getattr(sa, "__file__", None):  # replace stubs from test harness
    sys.modules.pop("sqlalchemy", None)
    sys.modules.pop("sqlalchemy.engine", None)
    sa = importlib.import_module("sqlalchemy")

from sqlalchemy import Column, Integer, MetaData, Table, Text, create_engine

from db_dedup import compute_content_hash, insert_if_unique


def test_compute_content_hash_order_independent():
    data1 = {"a": 1, "b": 2}
    data2 = {"b": 2, "a": 1}
    assert compute_content_hash(data1) == compute_content_hash(data2)


def test_insert_if_unique_duplicate_returns_existing_id(caplog):
    engine = create_engine("sqlite:///:memory:")
    meta = MetaData()
    tbl = Table(
        "items",
        meta,
        Column("id", Integer, primary_key=True),
        Column("name", Text),
        Column("content_hash", Text, unique=True),
    )
    meta.create_all(engine)

    with engine.begin() as conn:
        id1 = insert_if_unique(
            conn,
            tbl,
            {"name": "alpha"},
            ["name"],
            "m1",
        )
        assert id1 is not None

        caplog.clear()
        with caplog.at_level(logging.WARNING):
            id2 = insert_if_unique(
                conn,
                tbl,
                {"name": "alpha"},
                ["name"],
                "m1",
            )
        assert id2 == id1
        assert any("Duplicate insert ignored" in r.message for r in caplog.records)
