import builtins
import asyncio
import types
import os
import pytest
import sandbox_runner.environment as env

class DummyPopen:
    def __init__(self, *a, **k):
        if k.get("preexec_fn"):
            k["preexec_fn"]()
        self.pid = 1
        self.returncode = 0
        self.stdout = ""
        self.stderr = ""
    def communicate(self, timeout=None):
        return ("", "")
    def kill(self):
        pass


def test_execute_local_cgroup(monkeypatch, tmp_path):
    if not env._cgroup_v2_supported():
        pytest.skip("cgroup v2 not supported")

    base = tmp_path / "cgroup"
    base.mkdir()
    (base / "cgroup.controllers").write_text("cpu memory")

    monkeypatch.setattr(env, "_CGROUP_BASE", base)
    monkeypatch.setattr(env, "_cgroup_v2_supported", lambda: True)
    monkeypatch.setattr(env, "_rlimits_supported", lambda: False)
    monkeypatch.setattr(env, "_psutil_rlimits_supported", lambda: False)
    monkeypatch.setattr(env, "resource", None)
    monkeypatch.setattr(env, "psutil", None)
    monkeypatch.setattr(env.random, "randint", lambda a, b: 1)
    monkeypatch.setattr(env.os, "getpid", lambda: 42)

    orig_cleanup = env._cleanup_cgroup
    paths = []
    def fake_cleanup(p):
        paths.append(p)
    monkeypatch.setattr(env, "_cleanup_cgroup", fake_cleanup)

    orig_import = builtins.__import__
    def fake_import(name, *a, **k):
        if name == "docker":
            raise ImportError("no docker")
        return orig_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(env.subprocess, "Popen", DummyPopen)

    res = asyncio.run(env._execute_in_container("print('x')", {"CPU_LIMIT": "0.2", "MEMORY_LIMIT": "2Mi"}))
    assert res["exit_code"] == 0.0
    assert paths
    cg = paths[0]
    assert (cg / "cpu.max").read_text() == "20000 100000"
    assert (cg / "memory.max").read_text() == str(2 * 1024 * 1024)
    assert (cg / "cgroup.procs").read_text() == "42"
    orig_cleanup(cg)
    assert not cg.exists()
