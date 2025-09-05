import json
import sqlite3
import subprocess
import sys
import types
from types import SimpleNamespace
from dynamic_path_router import resolve_path  # noqa: F401

import menace_sandbox.sandbox_runner.scoring as scoring
import sandbox_results_logger as legacy

metric_stub = types.SimpleNamespace(
    inc=lambda *a, **k: None, labels=lambda **k: types.SimpleNamespace(inc=lambda *a, **k: None)
)
sys.modules.setdefault(
    "metrics_exporter",
    types.SimpleNamespace(
        sandbox_crashes_total=metric_stub,
        environment_failure_total=metric_stub,
        Gauge=lambda *a, **k: metric_stub,
    ),
)
sys.modules.setdefault(
    "menace_sandbox.sandbox_runner.environment",
    types.SimpleNamespace(get_edge_case_stubs=lambda: {}),
)


class _DummyParser:
    @staticmethod
    def parse(text):
        return {}


sys.modules.setdefault(
    "menace_sandbox.error_parser", types.SimpleNamespace(ErrorParser=_DummyParser)
)


def test_run_records_metrics(tmp_path, monkeypatch):
    log_dir = tmp_path / "logs"  # path-ignore
    monkeypatch.setattr(scoring, "_LOG_DIR", log_dir)
    monkeypatch.setattr(scoring, "_RUN_LOG", log_dir / "run_metrics.jsonl")  # path-ignore
    monkeypatch.setattr(scoring, "_SUMMARY_FILE", log_dir / "run_summary.json")  # path-ignore
    monkeypatch.setattr(legacy, "_LOG_DIR", log_dir)
    monkeypatch.setattr(legacy, "_DB_PATH", log_dir / "run_metrics.db")  # path-ignore

    result = SimpleNamespace(success=True, duration=1.0, failure=None)
    scoring.record_run(
        result,
        {
            "roi": 1.0,
            "coverage": {"file.py": ["func"]},  # path-ignore
            "entropy_delta": 0.1,
            "executed_functions": ["file.py:func"],  # path-ignore
        },
    )

    assert scoring._RUN_LOG.read_text().count("\n") == 1
    summary = json.loads(scoring._SUMMARY_FILE.read_text())
    assert summary["runs"] == 1

    conn = sqlite3.connect(log_dir / "run_metrics.db")  # noqa: SQL001  # path-ignore
    count = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    conn.close()
    assert count == 1


def test_run_tests_persists_metrics(tmp_path, monkeypatch):
    repo = tmp_path / "repo"  # path-ignore
    repo.mkdir()
    (repo / "dummy.txt").write_text("hi")  # path-ignore
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=repo, check=True)
    subprocess.run(["git", "add", "dummy.txt"], cwd=repo, check=True)  # path-ignore
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)

    log_dir = tmp_path / "logs"  # path-ignore
    monkeypatch.setattr(scoring, "_LOG_DIR", log_dir)
    monkeypatch.setattr(scoring, "_RUN_LOG", log_dir / "run_metrics.jsonl")  # path-ignore
    monkeypatch.setattr(scoring, "_SUMMARY_FILE", log_dir / "run_summary.json")  # path-ignore
    monkeypatch.setattr(scoring, "_db_record_run", lambda record: None)

    import menace_sandbox.sandbox_runner.test_harness as th

    dummy = th.TestHarnessResult(
        success=True,
        stdout="",
        stderr="",
        duration=0.1,
        failure=None,
        path=None,
        stub={},
        preset={},
        coverage={"file.py": ["func"], "executed_functions": ["file.py:func"]},  # path-ignore
        edge_cases=None,
        entropy_delta=0.5,
        executed_functions=["file.py:func"],  # path-ignore
    )
    monkeypatch.setattr(th, "_run_once", lambda *a, **k: dummy)

    res = th.run_tests(repo, input_stubs=[{}], presets=[{}])
    assert res is dummy

    lines = scoring._RUN_LOG.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["runtime"] == dummy.duration
    assert record["executed_functions"] == dummy.executed_functions
    assert record["entropy_delta"] == dummy.entropy_delta
