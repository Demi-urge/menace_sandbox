import json
import logging
import subprocess
import types
import importlib
import sys
from pathlib import Path

import dynamic_path_router
import sandbox_runner.environment as env


def _prepare_environment(monkeypatch, subprocess_run=None):
    class DummyErrorLogger:
        def __init__(self, *args, **kwargs):
            pass

    class DummyFileLock:
        def __init__(self, *args, **kwargs):
            self.path = args[0] if args else ""

        def acquire(self, *args, **kwargs):
            return True

        def release(self):
            return True

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setitem(
        sys.modules,
        "error_logger",
        types.SimpleNamespace(ErrorLogger=DummyErrorLogger),
    )
    dummy_timeout = type("Timeout", (Exception,), {})
    monkeypatch.setitem(
        sys.modules,
        "filelock",
        types.SimpleNamespace(FileLock=DummyFileLock, Timeout=dummy_timeout),
    )
    import lock_utils

    class DummySandboxLock:
        def __init__(self, *args, **kwargs):
            self.lock_file = args[0] if args else ""

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def acquire(self, *args, **kwargs):
            return True

        def release(self):
            return True

        @property
        def is_locked(self):
            return False

    monkeypatch.setattr(
        lock_utils,
        "SandboxLock",
        DummySandboxLock,
        raising=False,
    )
    monkeypatch.setattr(
        dynamic_path_router,
        "resolve_path",
        lambda value: Path(value),
        raising=False,
    )
    monkeypatch.setattr(
        dynamic_path_router,
        "repo_root",
        lambda: Path.cwd(),
        raising=False,
    )
    monkeypatch.setattr(
        dynamic_path_router,
        "path_for_prompt",
        lambda value: str(value),
        raising=False,
    )
    if subprocess_run is not None:
        monkeypatch.setattr(subprocess, "run", subprocess_run, raising=False)

    global env
    env = importlib.reload(env)
    if subprocess_run is not None:
        monkeypatch.setattr(env.subprocess, "run", subprocess_run, raising=False)


def test_unlabeled_volume_network_cleanup(monkeypatch, tmp_path):
    cmds = []

    def fake_run(cmd, stdout=None, stderr=None, text=None, check=False, timeout=None):
        cmds.append(cmd)
        if cmd[:3] == ["docker", "volume", "ls"]:
            return types.SimpleNamespace(returncode=0, stdout="vol1\n")
        if cmd[:3] == ["docker", "network", "ls"]:
            return types.SimpleNamespace(returncode=0, stdout="net1\n")
        if cmd[:3] == ["docker", "volume", "inspect"]:
            data = [{"CreatedAt": "1970-01-01T00:00:00Z", "Labels": None}]
            return types.SimpleNamespace(returncode=0, stdout=json.dumps(data))
        if cmd[:3] == ["docker", "network", "inspect"]:
            data = [{"Created": "1970-01-01T00:00:00Z", "Labels": None, "Name": "net1"}]
            return types.SimpleNamespace(returncode=0, stdout=json.dumps(data))
        return types.SimpleNamespace(returncode=0, stdout="")

    _prepare_environment(monkeypatch, subprocess_run=fake_run)
    active_containers = tmp_path / "active.json"
    monkeypatch.setattr(
        env,
        "_ACTIVE_CONTAINERS_FILE",
        active_containers,
        raising=False,
    )
    monkeypatch.setattr(
        env,
        "_ACTIVE_CONTAINERS_LOCK",
        env.FileLock(str(active_containers) + ".lock"),
        raising=False,
    )
    active_containers.write_text("[]")
    monkeypatch.setattr(
        env,
        "_ACTIVE_OVERLAYS_FILE",
        tmp_path / "overlays.json",
        raising=False,
    )
    monkeypatch.setattr(
        env,
        "_ACTIVE_OVERLAYS_LOCK",
        env.FileLock(str(tmp_path / "overlays.json") + ".lock"),
        raising=False,
    )
    env._write_active_overlays([])
    monkeypatch.setattr(env, "_purge_stale_vms", lambda: 0)

    monkeypatch.setattr(env, "_PRUNE_VOLUMES", True)
    monkeypatch.setattr(env, "_PRUNE_NETWORKS", True)
    monkeypatch.setattr(env, "_CONTAINER_MAX_LIFETIME", 1.0)
    monkeypatch.setattr(env.time, "time", lambda: 100.0)

    env._CLEANUP_METRICS.clear()

    env.purge_leftovers()

    assert ["docker", "volume", "rm", "-f", "vol1"] in cmds
    assert ["docker", "network", "rm", "-f", "net1"] in cmds
    assert env._CLEANUP_METRICS["volume"] == 1
    assert env._CLEANUP_METRICS["network"] == 1


def test_cleanup_artifacts_timeout(monkeypatch, caplog):
    _prepare_environment(monkeypatch)
    monkeypatch.setattr(env, "shutil", __import__("shutil"), raising=False)
    monkeypatch.setattr(env.shutil, "which", lambda name: True)

    recorded: dict[str, str | None] = {}

    def fake_record(item: str, *, reason: str | None = None) -> None:
        recorded[item] = reason

    monkeypatch.setattr(env, "_record_failed_cleanup", fake_record)

    calls: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout"))

    monkeypatch.setattr(env.subprocess, "run", fake_run)

    env._CLEANUP_METRICS.clear()
    baseline = env._CLEANUP_METRICS.copy()

    with caplog.at_level(logging.WARNING):
        env.cleanup_artifacts()

    assert calls == [["docker", "container", "prune", "-f"]]
    assert any("container prune timed out" in rec.message for rec in caplog.records)
    assert recorded == {"docker:container-prune": "docker container prune timeout"}
    assert env._CLEANUP_METRICS == baseline
