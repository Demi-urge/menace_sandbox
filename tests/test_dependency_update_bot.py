import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
import types

import menace.dependency_update_bot as dub


def _make_fake_run(outdated=None, fail_install=False, fail_tests=False):
    calls = []

    def fake_run(cmd, capture_output=False, text=False, check=False):
        calls.append(cmd)
        if cmd[:2] == ["pip", "list"] or cmd[1:3] == ["-m", "pip"] and "list" in cmd:
            data = outdated or []
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(data))
        if "install" in cmd:
            if fail_install:
                raise subprocess.CalledProcessError(1, cmd)
            return subprocess.CompletedProcess(cmd, 0)
        if "freeze" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="pkg==1.1\n")
        if "pytest" in cmd:
            if fail_tests:
                raise subprocess.CalledProcessError(1, cmd)
            return subprocess.CompletedProcess(cmd, 0)
        return subprocess.CompletedProcess(cmd, 0)

    return fake_run, calls


@contextmanager
def _dummy_env():
    yield "python"


def test_run_cycle_success(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    Path("pyproject.toml").write_text("[project]\ndependencies=['pkg']\n")
    fake_run, calls = _make_fake_run(outdated=[{"name": "pkg"}])
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(dub.DependencyUpdater, "_temp_env", lambda self, container_image=None: _dummy_env())
    updater = dub.DependencyUpdater()
    updater.run_cycle()
    assert ["python", "-m", "pip", "install", "-U", "pkg"] in calls
    assert ["python", "-m", "pytest", "-q"] in calls
    assert [sys.executable, "-m", "pip", "install", "pkg==1.1"] in calls


def test_run_cycle_install_fail(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    Path("pyproject.toml").write_text("[project]\ndependencies=['pkg']\n")
    fake_run, calls = _make_fake_run(outdated=[{"name": "pkg"}], fail_install=True)
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(dub.DependencyUpdater, "_temp_env", lambda self, container_image=None: _dummy_env())
    updater = dub.DependencyUpdater()
    updater.run_cycle()
    assert ["python", "-m", "pip", "install", "-U", "pkg"] in calls
    assert ["python", "-m", "pytest", "-q"] not in calls
    assert [sys.executable, "-m", "pip", "install", "pkg==1.1"] not in calls


def test_run_cycle_tests_fail(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    Path("pyproject.toml").write_text("[project]\ndependencies=['pkg']\n")
    fake_run, calls = _make_fake_run(outdated=[{"name": "pkg"}], fail_tests=True)
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(dub.DependencyUpdater, "_temp_env", lambda self, container_image=None: _dummy_env())
    updater = dub.DependencyUpdater()
    updater.run_cycle()
    assert ["python", "-m", "pip", "install", "-U", "pkg"] in calls
    assert ["python", "-m", "pytest", "-q"] in calls
    assert [sys.executable, "-m", "pip", "install", "pkg==1.1"] not in calls


def test_update_os(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    Path("pyproject.toml").write_text("[project]\ndependencies=['pkg']\n")
    fake_run, calls = _make_fake_run(outdated=[{"name": "pkg"}])
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(dub.DependencyUpdater, "_temp_env", lambda self, container_image=None: _dummy_env())
    updater = dub.DependencyUpdater()
    updater.run_cycle(update_os=True)
    assert ["apt-get", "update"] in calls


def test_temp_env_container(monkeypatch):
    calls = []
    def fake_run(cmd, check=False):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(subprocess, "run", fake_run)
    updater = dub.DependencyUpdater()
    with updater._temp_env(container_image="python:3.11") as py:
        assert "docker exec" in py
    assert any(cmd[0] == "docker" and cmd[1] == "run" for cmd in calls)
    assert any(cmd[0] == "docker" and cmd[1] == "rm" for cmd in calls)


def test_run_cycle_calls_orchestrator(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    Path("pyproject.toml").write_text("[project]\ndependencies=['pkg']\n")
    fake_run, calls = _make_fake_run(outdated=[{"name": "pkg"}])
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(dub.DependencyUpdater, "_temp_env", lambda self, container_image=None: _dummy_env())
    sent = []
    monkeypatch.setattr(dub, "requests", types.SimpleNamespace(post=lambda url, json=None, timeout=5: sent.append((url, json))))
    updater = dub.DependencyUpdater(orchestrator_url="http://orc")
    updater.run_cycle()
    assert sent and sent[0][0] == "http://orc/deploy"
