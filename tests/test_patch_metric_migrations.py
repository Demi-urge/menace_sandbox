
import sys
from pathlib import Path
import sqlite3
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.modules.setdefault("data_bot", types.SimpleNamespace(MetricsDB=object))

from menace_sandbox.vector_metrics_db import VectorMetricsDB
from menace_sandbox.code_database import PatchHistoryDB


def test_vector_metrics_db_migration(tmp_path):
    db_path = tmp_path / "vm_old.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE patch_metrics(patch_id TEXT)")
    conn.commit()
    conn.close()

    db = VectorMetricsDB(db_path)
    cols = {c[1] for c in db.conn.execute("PRAGMA table_info(patch_metrics)").fetchall()}
    required = {"patch_difficulty", "time_to_completion", "error_trace_count", "effort_estimate", "enhancement_score"}
    assert required <= cols


def test_patch_history_db_migration(tmp_path):
    db_path = tmp_path / "ph_old.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE patch_history(id INTEGER PRIMARY KEY, filename TEXT, description TEXT, ts REAL)")
    conn.commit()
    conn.close()

    db = PatchHistoryDB(db_path)
    conn = db.router.get_connection("patch_history")
    cols = {c[1] for c in conn.execute("PRAGMA table_info(patch_history)").fetchall()}
    required = {"patch_difficulty", "time_to_completion", "error_trace_count", "effort_estimate", "enhancement_score"}
    assert required <= cols
