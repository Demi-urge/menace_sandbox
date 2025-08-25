import importlib
import sqlite3
import sys
from types import ModuleType, SimpleNamespace
from pathlib import Path
from enum import Enum

# Avoid importing the heavy package __init__ by creating a lightweight package
root = Path(__file__).resolve().parents[1]
pkg = ModuleType("menace_sandbox")
pkg.__path__ = [str(root)]
sys.modules.setdefault("menace_sandbox", pkg)
sys.path.insert(0, str(root))

# Stub dependencies required by codex_db_helpers
for name, cls_name in [
    ("chatgpt_enhancement_bot", "EnhancementDB"),
    ("workflow_summary_db", "WorkflowSummaryDB"),
    ("discrepancy_db", "DiscrepancyDB"),
    ("task_handoff_bot", "WorkflowDB"),
]:
    mod = ModuleType(f"menace_sandbox.{name}")
    setattr(mod, cls_name, type(cls_name, (), {}))
    sys.modules[f"menace_sandbox.{name}"] = mod

scope_mod = ModuleType("menace_sandbox.scope_utils")

class Scope(str, Enum):
    LOCAL = "local"
    GLOBAL = "global"
    ALL = "all"

def build_scope_clause(table, scope, menace_id):
    return "", []

scope_mod.Scope = Scope
scope_mod.build_scope_clause = build_scope_clause
sys.modules["menace_sandbox.scope_utils"] = scope_mod

helpers = importlib.import_module("menace_sandbox.codex_db_helpers")


def _setup_enh_db(tmp_path, monkeypatch, rows):
    conn = sqlite3.connect(tmp_path / "enh.db")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE enhancements("  # id, summary, confidence, outcome_score, timestamp
        "id INTEGER PRIMARY KEY, summary TEXT, confidence REAL, "
        "outcome_score REAL, timestamp TEXT, source_menace_id TEXT)"
    )
    conn.executemany(
        "INSERT INTO enhancements(id, summary, confidence, outcome_score, timestamp, source_menace_id)"
        " VALUES (?,?,?,?,?,?)",
        rows,
    )
    conn.commit()

    class StubEnhDB:
        def __init__(self):
            self.conn = conn
            self.router = SimpleNamespace(menace_id="m0")

        def vector(self, rid):
            return [float(rid)]

    monkeypatch.setattr(helpers, "EnhancementDB", StubEnhDB)


def _setup_ws_db(tmp_path, monkeypatch, rows):
    conn = sqlite3.connect(tmp_path / "ws.db")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE workflow_summaries("  # workflow_id, summary, timestamp
        "workflow_id INTEGER PRIMARY KEY, summary TEXT, timestamp TEXT, "
        "source_menace_id TEXT)"
    )
    conn.executemany(
        "INSERT INTO workflow_summaries(workflow_id, summary, timestamp, source_menace_id)"
        " VALUES (?,?,?,?)",
        rows,
    )
    conn.commit()

    class StubWSDB:
        def __init__(self):
            self.conn = conn
            self.router = SimpleNamespace(menace_id="m1")

        def vector(self, wid):
            return [float(wid)]

    monkeypatch.setattr(helpers, "WorkflowSummaryDB", StubWSDB)


def _setup_disc_db(tmp_path, monkeypatch, rows):
    conn = sqlite3.connect(tmp_path / "disc.db")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE discrepancies("  # id, message, confidence, outcome_score, ts
        "id INTEGER PRIMARY KEY, message TEXT, metadata TEXT, confidence REAL, "
        "outcome_score REAL, ts TEXT, source_menace_id TEXT)"
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
            self.router = SimpleNamespace(menace_id="m2")

        def vector(self, rid):
            return [float(rid)]

    monkeypatch.setattr(helpers, "DiscrepancyDB", StubDiscDB)


def _setup_wf_db(tmp_path, monkeypatch, rows):
    conn = sqlite3.connect(tmp_path / "wf.db")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE workflows("  # id, workflow text, timestamp
        "id INTEGER PRIMARY KEY, workflow TEXT, timestamp TEXT, source_menace_id TEXT)"
    )
    conn.executemany(
        "INSERT INTO workflows(id, workflow, timestamp, source_menace_id) VALUES (?,?,?,?)",
        rows,
    )
    conn.commit()

    class StubWFDB:
        def __init__(self):
            self.conn = conn
            self.router = SimpleNamespace(menace_id="m3")

        def vector(self, wid):
            return [float(wid)]

    monkeypatch.setattr(helpers, "TaskWorkflowDB", StubWFDB)
    monkeypatch.setattr(helpers, "WorkflowDB", StubWFDB, raising=False)


def test_fetch_enhancements(monkeypatch, tmp_path):
    rows = [
        (1, "a", 0.2, 0.5, "2023-01-01", "A"),
        (2, "b", 0.8, 0.3, "2023-01-02", "B"),
    ]
    _setup_enh_db(tmp_path, monkeypatch, rows)

    all_rows = helpers.fetch_enhancements(sort_by="timestamp", limit=10)
    assert [s.content for s in all_rows] == ["b", "a"]

    top = helpers.fetch_enhancements(sort_by="confidence", limit=1)
    assert top[0].confidence == 0.8

    emb = helpers.fetch_enhancements(include_embeddings=True, limit=1)
    assert emb[0].embedding and len(emb[0].embedding) > 0


def test_fetch_summaries(monkeypatch, tmp_path):
    rows = [
        (1, "s1", "2023-01-01", "A"),
        (2, "s2", "2023-01-02", "B"),
    ]
    _setup_ws_db(tmp_path, monkeypatch, rows)

    all_rows = helpers.fetch_summaries(limit=10)
    assert [s.content for s in all_rows] == ["s2", "s1"]

    limited = helpers.fetch_summaries(limit=1)
    assert limited[0].content == "s2"
    assert limited[0].timestamp == "2023-01-02"

    emb = helpers.fetch_summaries(include_embeddings=True, limit=1)
    assert emb[0].embedding and len(emb[0].embedding) > 0


def test_fetch_discrepancies(monkeypatch, tmp_path):
    rows = [
        (1, "d1", "{}", 0.1, 0.9, "2023-01-01", "A"),
        (2, "d2", "{}", 0.9, 0.1, "2023-01-02", "B"),
    ]
    _setup_disc_db(tmp_path, monkeypatch, rows)

    all_rows = helpers.fetch_discrepancies(sort_by="timestamp", limit=10)
    assert [s.content for s in all_rows] == ["d2", "d1"]

    top = helpers.fetch_discrepancies(sort_by="outcome_score", limit=1)
    assert top[0].outcome_score == 0.9

    emb = helpers.fetch_discrepancies(include_embeddings=True, limit=1)
    assert emb[0].embedding and len(emb[0].embedding) > 0


def test_fetch_workflows(monkeypatch, tmp_path):
    rows = [
        (1, "w1", "2023-01-01", "A"),
        (2, "w2", "2023-01-02", "B"),
    ]
    _setup_wf_db(tmp_path, monkeypatch, rows)

    all_rows = helpers.fetch_workflows(sort_by="timestamp", limit=10)
    assert [s.content for s in all_rows] == ["w2", "w1"]

    fallback = helpers.fetch_workflows(sort_by="outcome_score", limit=2)
    assert [s.content for s in fallback] == ["w2", "w1"]

    emb = helpers.fetch_workflows(include_embeddings=True, limit=1)
    assert emb[0].embedding and len(emb[0].embedding) > 0


def test_aggregate_samples(monkeypatch, tmp_path):
    _setup_enh_db(tmp_path, monkeypatch, [(1, "a", 0.2, 0.5, "2023-01-02", "A")])
    _setup_ws_db(tmp_path, monkeypatch, [(1, "s1", "2023-01-01", "A")])
    _setup_disc_db(tmp_path, monkeypatch, [(1, "d1", "{}", 0.1, 0.9, "2023-01-03", "A")])
    _setup_wf_db(tmp_path, monkeypatch, [(1, "w1", "2023-01-01", "A")])

    results = helpers.aggregate_samples(sort_by="timestamp", limit=3)
    assert [s.content for s in results] == ["d1", "a", "s1"]
