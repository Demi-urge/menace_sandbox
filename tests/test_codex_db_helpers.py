import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import codex_db_helpers as helpers


def make_db(table: str, text_col: str, rows):
    connect = getattr(sqlite3, "connect")
    conn = connect(":memory:")
    conn.execute(
        f"""CREATE TABLE {table}(
            id INTEGER PRIMARY KEY,
            source_menace_id TEXT,
            {text_col} TEXT,
            score REAL,
            roi REAL,
            confidence REAL,
            ts INTEGER
        )"""
    )
    conn.executemany(
        f"""INSERT INTO {table}(
            id, source_menace_id, {text_col}, score, roi, confidence, ts
        ) VALUES (?,?,?,?,?,?,?)""",
        rows,
    )
    return SimpleNamespace(conn=conn, vector=Mock(side_effect=lambda i: [i]))


@pytest.mark.parametrize(
    "fetcher,table,text_col",
    [
        (helpers.fetch_enhancements, "enhancements", "summary"),
        (helpers.fetch_summaries, "workflow_summaries", "summary"),
        (helpers.fetch_discrepancies, "discrepancies", "message"),
        (helpers.fetch_workflow_history, "workflow_history", "details"),
    ],
)
def test_fetch_helpers_order_scope_limit_and_embeddings(
    monkeypatch, fetcher, table, text_col
):
    rows = [
        (1, "m1", "a", 0.3, 0.0, 0.6, 300),
        (2, "m2", "b", 0.9, 0.0, 0.4, 100),
        (3, "m1", "c", 0.5, 0.0, 0.9, 200),
    ]
    db = make_db(table, text_col, rows)

    scopes = []
    original = helpers.apply_scope_to_query

    def spy(query, scope, menace_id, *a, **kw):
        scopes.append(scope)
        return original(query, scope, menace_id, *a, **kw)

    monkeypatch.setattr(helpers, "apply_scope_to_query", spy)

    expectations = {
        "confidence": [3, 1],
        "score": [2, 3],
        "ts": [1, 3],
    }

    for sort_by, expected_ids in expectations.items():
        db.vector.reset_mock()
        result = fetcher(db, sort_by=sort_by, limit=2, with_embeddings=True)
        assert [r["id"] for r in result] == expected_ids
        assert len(result) == 2
        assert db.vector.call_count == 2
        assert all("embedding" in r for r in result)

    assert scopes == ["all"] * len(expectations)


def test_aggregate_training_samples_merges_and_sorts():
    enh_db = make_db(
        "enhancements", "summary", [(1, "m1", "e", 0.5, 0.0, 0.1, 10)]
    )
    sum_db = make_db(
        "workflow_summaries", "summary", [(1, "m2", "s", 0.9, 0.0, 0.2, 20)]
    )
    dis_db = make_db(
        "discrepancies", "message", [(1, "m1", "d", 0.7, 0.0, 0.3, 15)]
    )
    wf_db = make_db(
        "workflow_history", "details", [(1, "m2", "w", 0.8, 0.0, 0.4, 5)]
    )

    result = helpers.aggregate_training_samples(
        enh_db,
        sum_db,
        dis_db,
        wf_db,
        sort_by="score",
        limit=3,
        with_embeddings=True,
    )

    assert [r["score"] for r in result] == [0.9, 0.8, 0.7]
    assert len(result) == 3
    assert all("embedding" in r for r in result)
    assert enh_db.vector.call_count == 1
    assert sum_db.vector.call_count == 1
    assert dis_db.vector.call_count == 1
    assert wf_db.vector.call_count == 1
