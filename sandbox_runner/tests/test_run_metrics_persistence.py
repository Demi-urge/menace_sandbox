import json
import sqlite3
from types import SimpleNamespace

import sandbox_runner.scoring as scoring
import sandbox_results_logger as legacy


def test_run_records_metrics(tmp_path, monkeypatch):
    log_dir = tmp_path / "logs"
    monkeypatch.setattr(scoring, "_LOG_DIR", log_dir)
    monkeypatch.setattr(scoring, "_RUN_LOG", log_dir / "run_metrics.jsonl")
    monkeypatch.setattr(scoring, "_SUMMARY_FILE", log_dir / "run_summary.json")
    monkeypatch.setattr(legacy, "_LOG_DIR", log_dir)
    monkeypatch.setattr(legacy, "_DB_PATH", log_dir / "run_metrics.db")

    result = SimpleNamespace(success=True, duration=1.0, failure=None)
    scoring.record_run(
        result,
        {
            "roi": 1.0,
            "coverage": {"file.py": ["func"]},
            "entropy_delta": 0.1,
            "executed_functions": ["file.py:func"],
        },
    )

    assert scoring._RUN_LOG.read_text().count("\n") == 1
    summary = json.loads(scoring._SUMMARY_FILE.read_text())
    assert summary["runs"] == 1

    conn = sqlite3.connect(log_dir / "run_metrics.db")
    count = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    conn.close()
    assert count == 1

