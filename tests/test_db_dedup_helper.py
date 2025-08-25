import importlib
import logging
import sys

import pytest
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


def test_compute_content_hash_list_order_independent():
    data1 = {"a": [1, 2, 3]}
    data2 = {"a": [3, 2, 1]}
    assert compute_content_hash(data1) == compute_content_hash(data2)


def test_compute_content_hash_nested_list_order_independent():
    data1 = {"a": [{"x": 1}, {"x": 2}]}
    data2 = {"a": [{"x": 2}, {"x": 1}]}
    assert compute_content_hash(data1) == compute_content_hash(data2)


def test_insert_if_unique_duplicate_returns_existing_id(caplog):
    engine = create_engine("sqlite:///:memory:")
    meta = MetaData()
    tbl = Table(
        "items",
        meta,
        Column("id", Integer, primary_key=True),
        Column("name", Text),
        Column("content_hash", Text, unique=True, nullable=False),
    )
    meta.create_all(engine)
    logger = logging.getLogger(__name__)

    id1 = insert_if_unique(
        tbl,
        {"name": "alpha"},
        ["name"],
        "m1",
        engine=engine,
        logger=logger,
    )
    assert id1 is not None

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        id2 = insert_if_unique(
            tbl,
            {"name": "alpha"},
            ["name"],
            "m1",
            engine=engine,
            logger=logger,
        )
    assert id2 == id1
    with engine.begin() as conn:
        count = conn.execute(sa.select(sa.func.count()).select_from(tbl)).scalar()
    assert count == 1
    assert any("Duplicate insert ignored" in r.message for r in caplog.records)


def test_insert_if_unique_missing_field_sqlalchemy():
    engine = create_engine("sqlite:///:memory:")
    meta = MetaData()
    tbl = Table(
        "items",
        meta,
        Column("id", Integer, primary_key=True),
        Column("name", Text),
        Column("content_hash", Text, unique=True, nullable=False),
    )
    meta.create_all(engine)
    logger = logging.getLogger(__name__)

    with pytest.raises(KeyError, match="Missing fields for hashing: name"):
        insert_if_unique(
            tbl,
            {},
            ["name"],
            "m1",
            engine=engine,
            logger=logger,
        )
