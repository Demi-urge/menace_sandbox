import os
import sys
import asyncio
import types
from pathlib import Path
import sandbox_runner.environment as env


def _stub_docker(calls, tmp_dir):
    class DummyContainer:
        def __init__(self):
            self.id = "dummy"
        def wait(self, timeout=None):
            calls.append("wait")
            return {"StatusCode": 0}
        def logs(self, stdout=True, stderr=False):
            calls.append("logs")
            return b""
        def stats(self, stream=False):
            calls.append("stats")
            return {
                "blkio_stats": {"io_service_bytes_recursive": []},
                "cpu_stats": {"cpu_usage": {"total_usage": 1}},
                "memory_stats": {"max_usage": 1},
                "networks": {},
            }
        def remove(self, force=True):
            calls.append("remove")
    class DummyContainers:
        def run(self, image, cmd, **kwargs):
            calls.append("run")
            volumes = kwargs.get("volumes", {})
            if str(tmp_dir) in volumes:
                data = Path(tmp_dir) / "data"
                data.mkdir(exist_ok=True)
                (data / "roi_history.json").write_text("[]")
            return DummyContainer()
    class DummyClient:
        def __init__(self):
            self.containers = DummyContainers()
    mod = types.ModuleType("docker")
    mod.from_env = lambda: DummyClient()
    mod.types = types
    class DummyErr(Exception):
        pass
    err_mod = types.ModuleType("docker.errors")
    err_mod.DockerException = DummyErr
    err_mod.APIError = DummyErr
    mod.errors = err_mod
    sys.modules["docker"] = mod
    sys.modules["docker.errors"] = err_mod
    return mod, DummyErr


def _stub_docker_fail(tmp_dir):
    calls = []
    class DummyErr(Exception):
        pass
    class DummyContainers:
        def run(self, image, cmd, **kwargs):
            raise DummyErr("boom")
    class DummyClient:
        def __init__(self):
            self.containers = DummyContainers()
    mod = types.ModuleType("docker")
    mod.from_env = lambda: DummyClient()
    mod.types = types
    err_mod = types.ModuleType("docker.errors")
    err_mod.DockerException = DummyErr
    err_mod.APIError = DummyErr
    mod.errors = err_mod
    sys.modules["docker"] = mod
    sys.modules["docker.errors"] = err_mod
    return mod, DummyErr, calls

def test_simulate_full_environment_docker(monkeypatch, tmp_path):
    tmp_clone = tmp_path / "clone"
    tmp_clone.mkdir()
    orig_mkdtemp = env.tempfile.mkdtemp

    def fake_mkdtemp(*a, **k):
        if not getattr(fake_mkdtemp, "used", False):
            fake_mkdtemp.used = True
            return str(tmp_clone)
        return orig_mkdtemp(*a, **k)

    monkeypatch.setattr(env.tempfile, "mkdtemp", fake_mkdtemp)
    monkeypatch.setenv("SANDBOX_DOCKER", "1")
    monkeypatch.setattr(env, "_docker_available", lambda: True)

    calls = []
    docker_mod, DummyErr = _stub_docker(calls, tmp_clone)
    monkeypatch.setattr(env, "docker", docker_mod)
    monkeypatch.setattr(env, "DockerException", DummyErr)
    monkeypatch.setattr(env, "APIError", DummyErr)
    monkeypatch.setattr(env, "_DOCKER_CLIENT", None)

    monkeypatch.setattr(env, "_record_active_container", lambda cid: None)
    monkeypatch.setattr(env, "_remove_active_container", lambda cid: None)
    monkeypatch.setattr(env, "_register_container_finalizer", lambda c: None)
    env._CONTAINER_POOLS.clear()
    env._CONTAINER_DIRS.clear()
    env._CONTAINER_LAST_USED.clear()
    env._WARMUP_TASKS.clear()
    env._CLEANUP_TASK = None

    class DummyTracker:
        def __init__(self):
            self.loaded = None
        def load_history(self, path):
            self.loaded = path
    monkeypatch.setitem(sys.modules, "menace.roi_tracker", types.SimpleNamespace(ROITracker=DummyTracker))

    tracker = env.simulate_full_environment({})
    assert "run" in calls and "remove" in calls
    assert tracker.loaded == str(tmp_clone / "data" / "roi_history.json")
    assert not tmp_clone.exists()


def test_simulate_full_environment_docker_fallback(monkeypatch, tmp_path):
    tmp_clone = tmp_path / "clone2"
    tmp_clone.mkdir()
    orig_mkdtemp = env.tempfile.mkdtemp

    def fake_mkdtemp(*a, **k):
        if not getattr(fake_mkdtemp, "used", False):
            fake_mkdtemp.used = True
            return str(tmp_clone)
        return orig_mkdtemp(*a, **k)

    monkeypatch.setattr(env.tempfile, "mkdtemp", fake_mkdtemp)
    monkeypatch.setenv("SANDBOX_DOCKER", "1")
    monkeypatch.setattr(env, "_docker_available", lambda: True)

    docker_mod, DummyErr, calls = _stub_docker_fail(tmp_clone)
    monkeypatch.setattr(env, "docker", docker_mod)
    monkeypatch.setattr(env, "DockerException", DummyErr)
    monkeypatch.setattr(env, "APIError", DummyErr)
    monkeypatch.setattr(env, "_DOCKER_CLIENT", None)
    monkeypatch.setattr(env, "_CREATE_RETRY_LIMIT", 1)
    monkeypatch.setattr(env.time, "sleep", lambda s: None)
    monkeypatch.setattr(env, "_execute_in_container", lambda *a, **k: (_ for _ in ()).throw(DummyErr("boom")))

    monkeypatch.setattr(env, "_record_active_container", lambda cid: None)
    monkeypatch.setattr(env, "_remove_active_container", lambda cid: None)
    monkeypatch.setattr(env, "_register_container_finalizer", lambda c: None)
    env._CONTAINER_POOLS.clear()
    env._CONTAINER_DIRS.clear()
    env._CONTAINER_LAST_USED.clear()
    env._WARMUP_TASKS.clear()
    env._CLEANUP_TASK = None

    def fake_popen(cmd, **kwargs):
        class DummyProc:
            def communicate(self, timeout=None):
                return "", ""
            def kill(self):
                pass
            @property
            def returncode(self):
                return 0
        return DummyProc()
    def fake_run(cmd, **kwargs):
        if cmd[0] == "python":
            data = tmp_clone / "data"
            data.mkdir(exist_ok=True)
            (data / "roi_history.json").write_text("[]")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(env.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(env.subprocess, "run", fake_run)

    class DummyTracker:
        def __init__(self):
            self.loaded = None
            self.diagnostics = {}
        def load_history(self, path):
            self.loaded = path
    monkeypatch.setitem(sys.modules, "menace.roi_tracker", types.SimpleNamespace(ROITracker=DummyTracker))

    tracker = env.simulate_full_environment({})
    assert tracker.loaded == str(tmp_clone / "data" / "roi_history.json")
    assert tracker.diagnostics.get("local_execution") == "docker"


def test_simulate_full_environment_docker_retry(monkeypatch, tmp_path):
    tmp_clone = tmp_path / "retry"
    tmp_clone.mkdir()
    orig_mkdtemp = env.tempfile.mkdtemp

    def fake_mkdtemp(*a, **k):
        if not getattr(fake_mkdtemp, "used", False):
            fake_mkdtemp.used = True
            return str(tmp_clone)
        return orig_mkdtemp(*a, **k)

    monkeypatch.setattr(env.tempfile, "mkdtemp", fake_mkdtemp)
    monkeypatch.setenv("SANDBOX_DOCKER", "1")
    monkeypatch.setattr(env, "_docker_available", lambda: True)

    docker_mod, DummyErr = _stub_docker([], tmp_clone)
    monkeypatch.setattr(env, "docker", docker_mod)
    monkeypatch.setattr(env, "DockerException", DummyErr)
    monkeypatch.setattr(env, "APIError", DummyErr)
    monkeypatch.setattr(env, "_DOCKER_CLIENT", None)

    calls = 0

    def fake_exec(*a, **k):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise DummyErr("boom")
        data = tmp_clone / "data"
        data.mkdir(exist_ok=True)
        (data / "roi_history.json").write_text("[]")
        return {}

    delays = []
    monkeypatch.setattr(env.time, "sleep", lambda d: delays.append(d))
    monkeypatch.setattr(env, "_CREATE_RETRY_LIMIT", 3)
    monkeypatch.setattr(env, "_CREATE_BACKOFF_BASE", 1.0)
    monkeypatch.setattr(env, "_execute_in_container", fake_exec)

    monkeypatch.setattr(env, "_record_active_container", lambda cid: None)
    monkeypatch.setattr(env, "_remove_active_container", lambda cid: None)
    monkeypatch.setattr(env, "_register_container_finalizer", lambda c: None)

    class DummyTracker:
        def __init__(self):
            self.loaded = None
            self.diagnostics = {}
        def load_history(self, path):
            self.loaded = path

    monkeypatch.setitem(sys.modules, "menace.roi_tracker", types.SimpleNamespace(ROITracker=DummyTracker))

    tracker = env.simulate_full_environment({})
    assert calls == 3
    assert delays == [1.0, 2.0]
    assert tracker.loaded == str(tmp_clone / "data" / "roi_history.json")
    assert "local_execution" not in tracker.diagnostics


def test_pool_warmup_failure_metrics(monkeypatch):
    stub = types.ModuleType("metrics_exporter")

    class IncGauge:
        def __init__(self):
            self.called = []
        def labels(self, image):
            def inc():
                self.called.append(image)
            def set_val(v):
                pass
            return types.SimpleNamespace(inc=inc, set=set_val)

    stub.container_creation_failures_total = IncGauge()
    stub.container_creation_alerts_total = IncGauge()
    stub.container_creation_seconds = IncGauge()
    stub.container_creation_success_total = IncGauge()

    monkeypatch.setitem(sys.modules, "metrics_exporter", stub)
    monkeypatch.setitem(sys.modules, "sandbox_runner.metrics_exporter", stub)

    monkeypatch.setattr(env, "_DOCKER_CLIENT", object())
    env._CONTAINER_POOLS.clear()
    env._CONTAINER_DIRS.clear()
    env._CONTAINER_LAST_USED.clear()
    env._WARMUP_TASKS.clear()
    env._CREATE_FAILURES.clear()
    env._CONSECUTIVE_CREATE_FAILURES.clear()

    async def fail_create(image: str):
        raise RuntimeError("fail")

    monkeypatch.setattr(env, "_create_pool_container", fail_create)
    monkeypatch.setattr(env, "_schedule_coroutine", lambda c: asyncio.run(c))

    env._CONTAINER_POOL_SIZE = 1

    env._ensure_pool_size_async("img")

    assert stub.container_creation_failures_total.called == ["img"]
    assert not env._CONTAINER_POOLS.get("img")
