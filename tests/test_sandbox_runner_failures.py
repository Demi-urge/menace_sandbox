import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import pytest
from dynamic_path_router import resolve_path


def _load_test_harness(monkeypatch):
    path = resolve_path("sandbox_runner/test_harness.py")  # path-ignore
    spec = importlib.util.spec_from_file_location("menace.sandbox_runner.test_harness", path)
    th = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = th
    spec.loader.exec_module(th)
    monkeypatch.setattr(th, "_python_bin", lambda v: Path(sys.executable))
    return th


def _prepare_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "requirements.txt").write_text("")
    tests = repo / "tests"
    tests.mkdir()
    (tests / "test_mod.py").write_text("def test_ok():\n    assert True\n")  # path-ignore
    subprocess.run(["git", "init"], cwd=repo, capture_output=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, capture_output=True)
    return repo


def test_clone_failure_raises(monkeypatch):
    th = _load_test_harness(monkeypatch)

    def fake_run(cmd, *a, **kw):
        if cmd[:2] == ["git", "clone"]:
            return subprocess.CompletedProcess(cmd, 1, "", "clone err")
        pytest.fail(f"unexpected command {cmd}")

    monkeypatch.setattr(th.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError) as exc:
        th.run_tests(Path("does_not_matter"))
    assert "git clone failed" in str(exc.value)


def test_install_failure_raises(monkeypatch, tmp_path):
    th = _load_test_harness(monkeypatch)
    repo = _prepare_repo(tmp_path)

    def fake_run(cmd, *a, cwd=None, capture_output=None, text=None, check=None):
        if cmd[:2] == ["git", "clone"]:
            dst = Path(cmd[3])
            shutil.copytree(repo, dst, dirs_exist_ok=True)
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd[0] == sys.executable and cmd[1:3] == ["-m", "venv"]:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd[0] == sys.executable and cmd[1:3] == ["-m", "pip"]:
            return subprocess.CompletedProcess(cmd, 1, "", "install err")
        pytest.fail(f"unexpected command {cmd}")

    monkeypatch.setattr(th.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError) as exc:
        th.run_tests(repo)
    assert "dependency installation failed" in str(exc.value)
