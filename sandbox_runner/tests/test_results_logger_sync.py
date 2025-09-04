import os
import json
import sqlite3
from types import SimpleNamespace

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

import sandbox_runner.scoring as scoring
import sandbox_results_logger as legacy


def test_jsonl_and_sqlite_backends_receive_same_record(tmp_path, monkeypatch):
    monkeypatch.setattr(scoring, "_LOG_DIR", tmp_path)
    monkeypatch.setattr(scoring, "_RUN_LOG", tmp_path / "run_metrics.jsonl")
    monkeypatch.setattr(scoring, "_SUMMARY_FILE", tmp_path / "run_summary.json")

    monkeypatch.setattr(legacy, "_LOG_DIR", tmp_path)
    monkeypatch.setattr(legacy, "_DB_PATH", tmp_path / "run_metrics.db")

    scoring.record_run(SimpleNamespace(success=True, duration=1.0, failure=None), {})

    record = json.loads((tmp_path / "run_metrics.jsonl").read_text().splitlines()[0])

    conn = sqlite3.connect(tmp_path / "run_metrics.db")
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM runs").fetchone()
    sqlite_record = dict(row)
    for key in ("coverage", "executed_functions"):
        if sqlite_record[key] is not None:
            sqlite_record[key] = json.loads(sqlite_record[key])
    conn.close()

    assert sqlite_record == record
