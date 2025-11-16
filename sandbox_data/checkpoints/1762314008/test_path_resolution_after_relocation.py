import os
import sys
import types
from pathlib import Path

sys.modules.pop("dynamic_path_router", None)
from dynamic_path_router import clear_cache, resolve_path  # noqa: F401,E402
from sandbox_runner.edge_case_generator import generate_edge_cases  # noqa: E402
from sandbox_runner.metrics_plugins import discover_metrics_plugins  # noqa: E402
from sandbox_runner.input_history_db import InputHistoryDB  # noqa: E402
from sandbox_runner.workflow_sandbox_runner import WorkflowSandboxRunner  # noqa: E402


def test_edge_case_file_resolves_after_repo_move(tmp_path, monkeypatch):
    repo = tmp_path / "repo"  # path-ignore
    repo.mkdir()
    cfg = repo / "payloads.json"  # path-ignore
    cfg.write_text('{"extra.txt": "data"}', encoding="utf-8")  # path-ignore
    clear_cache()
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))
    monkeypatch.setenv("SANDBOX_HOSTILE_PAYLOADS_FILE", "payloads.json")  # path-ignore
    cases = generate_edge_cases()
    assert "extra.txt" in cases


def test_metrics_plugins_config_resolves_after_repo_move(tmp_path, monkeypatch):
    repo = tmp_path / "repo"  # path-ignore
    repo.mkdir()
    plugin_dir = repo / "plugins"  # path-ignore
    plugin_dir.mkdir()
    (plugin_dir / "p.py").write_text(  # path-ignore
        "def collect_metrics(prev, cur, res):\n    return {'x': 1}\n",
        encoding="utf-8",
    )
    cfg = repo / "metrics.yaml"  # path-ignore
    cfg.write_text("plugin_dirs:\n  - plugins\n", encoding="utf-8")  # path-ignore
    clear_cache()
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))
    monkeypatch.setenv("SANDBOX_METRICS_FILE", "metrics.yaml")  # path-ignore
    plugins = discover_metrics_plugins(os.environ)
    assert plugins and plugins[0](0.0, 0.0, None) == {"x": 1}


def test_input_history_db_uses_repo_path(tmp_path, monkeypatch):
    repo = tmp_path / "repo"  # path-ignore
    repo.mkdir()
    clear_cache()
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))

    class DummyConn:
        def execute(self, *a, **k):
            return self

        def fetchall(self):
            return []

        def commit(self):
            pass

    class DummyRouter:
        def get_connection(self, name):
            return DummyConn()

    monkeypatch.setattr("sandbox_runner.input_history_db.router", DummyRouter())
    db = InputHistoryDB()
    assert db.path == repo / "input_history.db"  # path-ignore


def test_workflow_runner_resolves_coverage_file(tmp_path, monkeypatch):
    repo = tmp_path / "repo"  # path-ignore
    repo.mkdir()
    clear_cache()
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))
    monkeypatch.setenv("SANDBOX_CAPTURE_COVERAGE", "1")
    monkeypatch.setenv("SANDBOX_COVERAGE_FILE", "cov.json")  # path-ignore

    called: dict[str, str] = {}

    class DummyData:
        def measured_files(self):
            return []

        def lines(self, f):
            return []

    class DummyCov:
        def __init__(self, *a, **kw):
            pass

        def start(self):
            pass

        def stop(self):
            pass

        def json_report(self, outfile):
            called["outfile"] = outfile

        def get_data(self):
            return DummyData()

    monkeypatch.setattr(
        "sandbox_runner.workflow_sandbox_runner.coverage",
        types.SimpleNamespace(Coverage=lambda *a, **kw: DummyCov(*a, **kw)),
    )

    runner = WorkflowSandboxRunner()
    runner.run(lambda: None, safe_mode=False, use_subprocess=False)
    assert Path(called["outfile"]) == repo / "cov.json"  # path-ignore
