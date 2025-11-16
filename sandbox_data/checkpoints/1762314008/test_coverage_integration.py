import os
import json
import subprocess

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
from menace_sandbox.sandbox_runner.workflow_sandbox_runner import (  # noqa: E402
    WorkflowSandboxRunner,
)
from menace_sandbox.sandbox_runner.test_harness import _run_once  # noqa: E402
import menace_sandbox.sandbox_runner.scoring as scoring  # noqa: E402


def _sample_module():
    def inner():
        return 42
    return inner()


def test_module_coverage_reporting(monkeypatch):
    monkeypatch.setenv("SANDBOX_CAPTURE_COVERAGE", "1")
    runner = WorkflowSandboxRunner()
    metrics = runner.run(_sample_module, use_subprocess=False)
    mod = metrics.modules[0]
    assert isinstance(mod.coverage_files, list)
    assert isinstance(mod.coverage_functions, list)


def test_harness_logs_function_coverage(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    mod_name = "mod" + ".py"
    test_name = "test_mod" + ".py"
    (repo / mod_name).write_text("def foo():\n    return 1\n", encoding="utf-8")
    (repo / test_name).write_text(
        "from mod import foo\n\n\ndef test_foo():\n    assert foo() == 1\n",
        encoding="utf-8",
    )
    (repo / "requirements.txt").write_text("pytest\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    env = {
        "GIT_AUTHOR_NAME": "a",
        "GIT_AUTHOR_EMAIL": "a@b.c",
        "GIT_COMMITTER_NAME": "a",
        "GIT_COMMITTER_EMAIL": "a@b.c",
    }
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, env=env)

    log_dir = tmp_path / "logs"
    monkeypatch.setattr(scoring, "_LOG_DIR", log_dir)
    monkeypatch.setattr(scoring, "_RUN_LOG", log_dir / "run_metrics.jsonl")
    monkeypatch.setattr(scoring, "_SUMMARY_FILE", log_dir / "run_summary.json")
    monkeypatch.delenv("SANDBOX_CAPTURE_COVERAGE", raising=False)

    res = _run_once(repo)
    assert res.coverage is not None
    assert isinstance(res.coverage.get("executed_functions"), list)
    data = json.loads(scoring._RUN_LOG.read_text().splitlines()[-1])
    assert data["functions_hit"] is not None
    assert isinstance(data["executed_functions"], list)
