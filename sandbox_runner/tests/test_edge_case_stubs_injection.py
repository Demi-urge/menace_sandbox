import json
import os
import subprocess
import sys
from pathlib import Path
import types

import pytest

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

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

from menace_sandbox.sandbox_runner.test_harness import run_tests


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
    test_file = repo / "test_edge_cases.py"
    test_file.write_text(
        """
import json
import os
import pathlib


def test_edge_cases_written():
    data = json.loads(os.environ['SANDBOX_EDGE_CASE_STUBS'])
    for name, content in data.items():
        p = pathlib.Path(name)
        assert p.exists()
        expected = content if isinstance(content, str) else json.dumps(content)
        assert p.read_text() == expected
""",
        encoding="utf-8",
    )
    (repo / "requirements.txt").write_text("pytest\n", encoding="utf-8")
    _git(["git", "init"], repo)
    _git(["git", "add", "test_edge_cases.py", "requirements.txt"], repo)
    _git(["git", "commit", "-m", "init"], repo)

    stubs = {"alpha.txt": "beta", "null.txt": None}
    env_stub = types.SimpleNamespace(get_edge_case_stubs=lambda: stubs)
    monkeypatch.setitem(sys.modules, "menace_sandbox.sandbox_runner.environment", env_stub)
    result = run_tests(repo, input_stubs=[{}], presets=[{}])
    assert result.success
