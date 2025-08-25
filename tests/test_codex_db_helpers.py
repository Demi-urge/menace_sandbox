import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock
import types
import sys

import pytest

for name, cls in [("chatgpt_enhancement_bot","EnhancementDB"),("workflow_summary_db","WorkflowSummaryDB"),("discrepancy_db","DiscrepancyDB"),("evolution_history_db","EvolutionHistoryDB")]:
    mod = types.ModuleType(name)
    setattr(mod, cls, object)
    sys.modules[name] = mod

import codex_db_helpers as helpers
from scope_utils import Scope



def make_enhancement_db():
    rows = [
        (1, "enh1", 0.3, "300", 0.1, 0.9),
        (2, "enh2", 0.9, "100", 0.2, 0.6),
        (3, "enh3", 0.5, "200", 0.3, 0.8),
    ]
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE enhancements(
        id INTEGER PRIMARY KEY,
        summary TEXT,
        score REAL,
        timestamp TEXT,
        roi REAL,
        confidence REAL
    )"""
    )
    conn.executemany(
        "INSERT INTO enhancements(id, summary, score, timestamp, roi, confidence) VALUES (?,?,?,?,?,?)",
        rows,
    )
    return SimpleNamespace(conn=conn, vector=Mock(side_effect=lambda i: [i]))



def make_workflow_summary_db():
    rows = [
        (1, "ws1", 0.1, 0.7, "100"),
        (3, "ws3", 0.2, 0.5, "300"),
        (2, "ws2", 0.3, 0.6, "200"),
    ]
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE workflow_summaries(
        workflow_id INTEGER PRIMARY KEY,
        summary TEXT,
        roi REAL,
        confidence REAL,
        ts TEXT
    )"""
    )
    conn.executemany(
        "INSERT INTO workflow_summaries(workflow_id, summary, roi, confidence, ts) VALUES (?,?,?,?,?)",
        rows,
    )
    return SimpleNamespace(conn=conn)



def make_discrepancy_db():
    rows = [
        (1, "d1", json.dumps({"outcome_score": 0.2, "confidence": 0.9}), "200"),
        (2, "d2", json.dumps({"outcome_score": 0.8, "confidence": 0.5}), "100"),
        (3, "d3", json.dumps({"outcome_score": 0.6, "confidence": 0.8}), "300"),
    ]
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE discrepancies(
        id INTEGER PRIMARY KEY,
        message TEXT,
        metadata TEXT,
        ts TEXT
    )"""
    )
    conn.executemany(
        "INSERT INTO discrepancies(id, message, metadata, ts) VALUES (?,?,?,?)",
        rows,
    )
    return SimpleNamespace(conn=conn, vector=Mock(side_effect=lambda i: [i]))



def make_evolution_db():
    rows = [
        ("e1", 0.4, 0.1, "100", 0, 0, 0, 0, "", "", "", "", "", "", 0, "", ""),
        ("e2", 0.7, 0.2, "300", 0, 0, 0, 0, "", "", "", "", "", "", 0, "", ""),
        ("e3", None, 0.9, "200", 0, 0, 0, 0, "", "", "", "", "", "", 0, "", ""),
    ]
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE evolution_history(
        action TEXT,
        roi REAL,
        performance REAL,
        ts TEXT,
        before_metric REAL,
        after_metric REAL,
        predicted_roi REAL,
        efficiency REAL,
        bottleneck TEXT,
        patch_id TEXT,
        workflow_id TEXT,
        trending_topic TEXT,
        reason TEXT,
        trigger TEXT,
        parent_event_id INTEGER,
        predicted_class TEXT,
        actual_class TEXT
    )"""
    )
    conn.executemany(
        "INSERT INTO evolution_history(action, roi, performance, ts, before_metric, after_metric, predicted_roi, efficiency, bottleneck, patch_id, workflow_id, trending_topic, reason, trigger, parent_event_id, predicted_class, actual_class) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    return SimpleNamespace(conn=conn, vector=Mock(side_effect=lambda i: [i]))



@pytest.mark.parametrize(
    "fetcher_name,db_factory,db_attr,expectations",
    [
        (
            "fetch_enhancement_samples",
            make_enhancement_db,
            "EnhancementDB",
            {
                "timestamp": ["enh1", "enh3"],
                "outcome_score": ["enh2", "enh3"],
            },
        ),
        (
            "fetch_workflow_summary_samples",
            make_workflow_summary_db,
            "WorkflowSummaryDB",
            {"timestamp": ["ws3", "ws2"]},
        ),
        (
            "fetch_discrepancy_samples",
            make_discrepancy_db,
            "DiscrepancyDB",
            {
                "timestamp": ["d3", "d1"],
                "outcome_score": ["d2", "d3"],
                "confidence": ["d1", "d3"],
            },
        ),
        (
            "fetch_evolution_samples",
            make_evolution_db,
            "EvolutionHistoryDB",
            {
                "timestamp": ["e2", "e3"],
                "outcome_score": ["e3", "e2"],
            },
        ),
    ],
)
def test_fetch_helpers_scope_sorting_and_vectors(monkeypatch, fetcher_name, db_factory, db_attr, expectations):
    db = db_factory()
    monkeypatch.setattr(helpers, db_attr, lambda: db)

    scopes: list = []

    def fake_apply_scope(query, scope, menace_id, table_alias=None, params=None):
        scopes.append(scope)
        return query, list(params) if params else []

    monkeypatch.setattr(helpers, "apply_scope_to_query", fake_apply_scope)

    fetcher = getattr(helpers, fetcher_name)

    for sort_by, expected_texts in expectations.items():
        if hasattr(db, "vector"):
            db.vector.reset_mock()
        result = fetcher(limit=2, sort_by=sort_by, with_vectors=True)
        assert [s.text for s in result] == expected_texts
        if hasattr(db, "vector"):
            assert db.vector.call_count == len(expected_texts)
            assert all(s.vector is not None for s in result)
        else:
            assert all(s.vector is None for s in result)

    assert scopes == [Scope.ALL] * len(expectations)


def test_aggregate_samples_merges_and_sorts(monkeypatch):
    enh_db = make_enhancement_db()
    wf_db = make_workflow_summary_db()
    dis_db = make_discrepancy_db()
    evo_db = make_evolution_db()

    monkeypatch.setattr(helpers, "EnhancementDB", lambda: enh_db)
    monkeypatch.setattr(helpers, "WorkflowSummaryDB", lambda: wf_db)
    monkeypatch.setattr(helpers, "DiscrepancyDB", lambda: dis_db)
    monkeypatch.setattr(helpers, "EvolutionHistoryDB", lambda: evo_db)

    scopes: list = []

    def fake_apply_scope(query, scope, menace_id, table_alias=None, params=None):
        scopes.append(scope)
        return query, list(params) if params else []

    monkeypatch.setattr(helpers, "apply_scope_to_query", fake_apply_scope)

    result = helpers.aggregate_samples(
        ["enhancement", "workflow_summary", "discrepancy", "evolution"],
        limit_per_source=2,
        sort_by="timestamp",
        with_vectors=True,
    )

    assert len(result) == 8
    ts_values = [s.ts or "" for s in result]
    assert ts_values == sorted(ts_values, reverse=True)
    assert enh_db.vector.call_count == 2
    assert dis_db.vector.call_count == 2
    assert evo_db.vector.call_count == 2
    for sample in result:
        if sample.source == "workflow_summary":
            assert sample.vector is None
        else:
            assert sample.vector is not None
    assert scopes == [Scope.ALL] * 4

