import os
import subprocess
import sys
from pathlib import Path
import types
from dynamic_path_router import get_project_root

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
sys.path.insert(0, str(get_project_root()))
root = get_project_root()
data_dir = root / "sandbox_data"
data_dir.mkdir(exist_ok=True)
(data_dir / "cleanup.log").write_text("", encoding="utf-8")


class _DummyLogger:
    def __init__(self, **_kwargs):
        pass

    def log(self, *_args, **_kwargs):
        pass


sys.modules.setdefault("error_logger", types.SimpleNamespace(ErrorLogger=_DummyLogger))

metric_stub = types.SimpleNamespace(
    inc=lambda: None, labels=lambda **k: types.SimpleNamespace(inc=lambda: None)
)
sys.modules.setdefault(
    "metrics_exporter",
    types.SimpleNamespace(
        sandbox_crashes_total=metric_stub,
        environment_failure_total=metric_stub,
        Gauge=lambda *a, **k: metric_stub,
    ),
)

import menace_sandbox.sandbox_runner.test_harness as th


def _git(cmd, cwd):
    subprocess.run(cmd, cwd=cwd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                   env={**os.environ,
                        "GIT_AUTHOR_NAME": "a",
                        "GIT_AUTHOR_EMAIL": "a@a",
                        "GIT_COMMITTER_NAME": "a",
                        "GIT_COMMITTER_EMAIL": "a@a"})


def test_edge_case_stubs_injected(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    name = "test_edge_cases" + "." + "py"
    test_file = repo / name
    test_file.write_text(
        """
import json
import os


def test_edge_cases_serialized(hostile_payloads):
    data = json.loads(os.environ['SANDBOX_EDGE_CASES'])
    assert hostile_payloads == data
""",
        encoding="utf-8",
    )
    (repo / "requirements.txt").write_text("pytest\n", encoding="utf-8")
    _git(["git", "init"], repo)
    _git(["git", "add", name, "requirements.txt"], repo)
    _git(["git", "commit", "-m", "init"], repo)

    stubs = {"alpha.txt": "beta", "null.txt": None, "http://edge-case.test/data": "payload"}
    monkeypatch.setattr(th, "get_edge_case_stubs", lambda: stubs)
    result = th.run_tests(repo, input_stubs=[{}], presets=[{}])
    assert result.success
    assert result.edge_cases == stubs
    for name in stubs:
        assert not (repo / name).exists()


def test_inject_edge_cases_flag(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    name = "test_edge_cases" + "." + "py"
    test_file = repo / name
    test_file.write_text(
        """
from pathlib import Path


def test_no_edge_cases_written():
    assert not Path('alpha.txt').exists()
""",
        encoding="utf-8",
    )
    (repo / "requirements.txt").write_text("pytest\n", encoding="utf-8")
    _git(["git", "init"], repo)
    _git(["git", "add", name, "requirements.txt"], repo)
    _git(["git", "commit", "-m", "init"], repo)

    stubs = {"alpha.txt": "beta"}
    monkeypatch.setattr(th, "get_edge_case_stubs", lambda: stubs)
    monkeypatch.setattr(
        th,
        "SandboxSettings",
        lambda: types.SimpleNamespace(inject_edge_cases=False),
    )
    result = th.run_tests(repo, input_stubs=[{}], presets=[{}])
    assert result.success
