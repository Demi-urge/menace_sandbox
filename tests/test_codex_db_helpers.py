import sqlite3
from types import SimpleNamespace, ModuleType
from pathlib import Path
import sys
import importlib
from enum import Enum

# Avoid executing the heavy package __init__ by inserting a lightweight
# placeholder package for imports.
root = Path(__file__).resolve().parents[1]
pkg = ModuleType("menace_sandbox")
pkg.__path__ = [str(root)]
sys.modules.setdefault("menace_sandbox", pkg)
sys.path.insert(0, str(root))

# Provide lightweight stub modules required by codex_db_helpers to avoid
# importing heavy dependencies during testing.
for name, cls_name in [
    ("chatgpt_enhancement_bot", "EnhancementDB"),
    ("workflow_summary_db", "WorkflowSummaryDB"),
    ("discrepancy_db", "DiscrepancyDB"),
    ("evolution_history_db", "EvolutionHistoryDB"),
]:
    mod = ModuleType(f"menace_sandbox.{name}")
    setattr(mod, cls_name, type(cls_name, (), {}))
    sys.modules[f"menace_sandbox.{name}"] = mod

scope_mod = ModuleType("menace_sandbox.scope_utils")

class Scope(str, Enum):
    LOCAL = "local"
    GLOBAL = "global"
    ALL = "all"


def build_scope_clause(table_name, scope, menace_id):
    return "", []


scope_mod.Scope = Scope
scope_mod.build_scope_clause = build_scope_clause
sys.modules["menace_sandbox.scope_utils"] = scope_mod

helpers = importlib.import_module("menace_sandbox.codex_db_helpers")


def _setup_enh_db(tmp_path, monkeypatch, rows):
    conn = sqlite3.connect(tmp_path / 'enh.db')
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE enhancements("
        "id INTEGER PRIMARY KEY,"
        "summary TEXT,"
        "confidence REAL,"
        "outcome_score REAL,"
        "timestamp TEXT,"
        "source_menace_id TEXT)"
    )
    conn.executemany(
        "INSERT INTO enhancements(id, summary, confidence, outcome_score, timestamp, source_menace_id)"
        " VALUES (?,?,?,?,?,?)",
        rows,
    )
    conn.commit()

    class StubEnhancementDB:
        def __init__(self):
            self.conn = conn
            self.router = SimpleNamespace(menace_id='m0')

        def vector(self, rid):
            return [float(rid)]

    monkeypatch.setattr(helpers, 'EnhancementDB', StubEnhancementDB)


def _setup_ws_db(tmp_path, monkeypatch, rows):
    conn = sqlite3.connect(tmp_path / 'ws.db')
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE workflow_summaries("
        "workflow_id INTEGER PRIMARY KEY,"
        "summary TEXT,"
        "source_menace_id TEXT)"
    )
    conn.executemany(
        "INSERT INTO workflow_summaries(workflow_id, summary, source_menace_id)"
        " VALUES (?,?,?)",
        rows,
    )
    conn.commit()

    class StubWSDB:
        def __init__(self):
            self.conn = conn
            self.router = SimpleNamespace(menace_id='m1')

        def vector(self, wid):
            return [float(wid)]

    monkeypatch.setattr(helpers, 'WorkflowSummaryDB', StubWSDB)


def _setup_disc_db(tmp_path, monkeypatch, rows):
    conn = sqlite3.connect(tmp_path / 'disc.db')
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE discrepancies("
        "id INTEGER PRIMARY KEY,"
        "message TEXT,"
        "metadata TEXT,"
        "confidence REAL,"
        "outcome_score REAL,"
        "ts TEXT,"
        "source_menace_id TEXT)"
    )
    conn.executemany(
        "INSERT INTO discrepancies(id, message, metadata, confidence, outcome_score, ts, source_menace_id)"
        " VALUES (?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()

    class StubDiscDB:
        def __init__(self):
            self.conn = conn
            self.router = SimpleNamespace(menace_id='m2')

        def vector(self, rid):
            return [float(rid)]

    monkeypatch.setattr(helpers, 'DiscrepancyDB', StubDiscDB)


def _setup_hist_db(tmp_path, monkeypatch, rows):
    conn = sqlite3.connect(tmp_path / 'hist.db')
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE evolution_history("
        "action TEXT,"
        "roi REAL,"
        "performance REAL,"
        "ts TEXT,"
        "source_menace_id TEXT)"
    )
    conn.executemany(
        "INSERT INTO evolution_history(action, roi, performance, ts, source_menace_id)"
        " VALUES (?,?,?,?,?)",
        rows,
    )
    conn.commit()

    class StubHistDB:
        def __init__(self):
            self.conn = conn
            self.router = SimpleNamespace(menace_id='m3')

        def vector(self, rid):
            return [float(rid)]

    monkeypatch.setattr(helpers, 'EvolutionHistoryDB', StubHistDB)


def test_fetch_enhancements(monkeypatch, tmp_path):
    rows = [
        (1, 'a', 0.2, 0.5, '2023-01-01T00:00:00', 'A'),
        (2, 'b', 0.8, 0.3, '2023-01-02T00:00:00', 'B'),
    ]
    _setup_enh_db(tmp_path, monkeypatch, rows)

    all_rows = helpers.fetch_enhancements(order_by='timestamp', limit=10)
    assert len(all_rows) == 2
    assert [ex.metadata['id'] for ex in all_rows] == [2, 1]

    top = helpers.fetch_enhancements(order_by='confidence', limit=1)
    assert len(top) == 1 and top[0].confidence == 0.8

    emb = helpers.fetch_enhancements(include_embeddings=True, limit=1)
    assert emb[0].embedding and len(emb[0].embedding) > 0


def test_fetch_workflow_summaries(monkeypatch, tmp_path):
    rows = [
        (1, 's1', 'A'),
        (2, 's2', 'B'),
    ]
    _setup_ws_db(tmp_path, monkeypatch, rows)

    all_rows = helpers.fetch_workflow_summaries(limit=10)
    assert len(all_rows) == 2
    assert [ex.metadata['workflow_id'] for ex in all_rows] == [2, 1]

    limited = helpers.fetch_workflow_summaries(limit=1)
    assert len(limited) == 1 and limited[0].metadata['workflow_id'] == 2

    emb = helpers.fetch_workflow_summaries(include_embeddings=True, limit=1)
    assert emb[0].embedding and len(emb[0].embedding) > 0


def test_fetch_discrepancies(monkeypatch, tmp_path):
    rows = [
        (1, 'd1', '{}', 0.1, 0.9, '2023-01-01', 'A'),
        (2, 'd2', '{}', 0.9, 0.1, '2023-01-02', 'B'),
    ]
    _setup_disc_db(tmp_path, monkeypatch, rows)

    all_rows = helpers.fetch_discrepancies(order_by='timestamp', limit=10)
    assert len(all_rows) == 2
    assert [ex.metadata['id'] for ex in all_rows] == [2, 1]

    top = helpers.fetch_discrepancies(order_by='outcome_score', limit=1)
    assert len(top) == 1 and top[0].outcome_score == 0.9

    emb = helpers.fetch_discrepancies(include_embeddings=True, limit=1)
    assert emb[0].embedding and len(emb[0].embedding) > 0


def test_fetch_workflow_history(monkeypatch, tmp_path):
    rows = [
        ('h1', 1.0, None, '2023-01-01', 'A'),
        ('h2', None, 2.0, '2023-01-02', 'B'),
    ]
    _setup_hist_db(tmp_path, monkeypatch, rows)

    all_rows = helpers.fetch_workflow_history(order_by='timestamp', limit=10)
    assert len(all_rows) == 2
    assert [ex.metadata['id'] for ex in all_rows] == [2, 1]

    top = helpers.fetch_workflow_history(order_by='outcome_score', limit=1)
    assert len(top) == 1 and top[0].outcome_score == 2.0

    emb = helpers.fetch_workflow_history(include_embeddings=True, limit=1)
    assert emb[0].embedding and len(emb[0].embedding) > 0


def test_aggregate_examples(monkeypatch, tmp_path):
    _setup_enh_db(tmp_path, monkeypatch, [(1, 'a', 0.2, 0.5, '2023-01-02', 'A')])
    _setup_ws_db(tmp_path, monkeypatch, [(1, 's1', 'A')])
    _setup_disc_db(tmp_path, monkeypatch, [(1, 'd1', '{}', 0.1, 0.9, '2023-01-03', 'A')])
    _setup_hist_db(tmp_path, monkeypatch, [('h1', 1.0, None, '2023-01-01', 'A')])

    results = helpers.aggregate_examples(order_by='timestamp', limit=3)
    timestamps = [ex.timestamp for ex in results]
    assert timestamps == ['2023-01-03', '2023-01-02', '2023-01-01']
