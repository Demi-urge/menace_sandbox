import importlib.util
import pathlib
import types
import sys
from dynamic_path_router import resolve_path

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "sandbox_runner.environment",
    resolve_path("sandbox_runner/environment.py"),
)
env = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.path.insert(0, str(ROOT))
pkg = types.ModuleType("sandbox_runner")
pkg.__path__ = [str(ROOT / "sandbox_runner")]
sys.modules.setdefault("sandbox_runner", pkg)
oi = types.ModuleType("sandbox_runner.orphan_integration")
oi.integrate_and_graph_orphans = lambda *a, **k: None
sys.modules["sandbox_runner.orphan_integration"] = oi
sys.modules["sandbox_runner.environment"] = env
spec.loader.exec_module(env)


class DummyLogger:
    def __init__(self):
        self.records: list[tuple[str, str]] = []

    def info(self, msg, *args, **kwargs):
        self.records.append(("info", msg % args if args else msg))

    def error(self, msg, *args, **kwargs):
        self.records.append(("error", msg % args if args else msg))

    def exception(self, msg, *args, **kwargs):
        self.records.append(("exception", msg % args if args else msg))


def test_docker_ps_failure_logged(monkeypatch):
    dummy_logger = DummyLogger()

    class DummyProc:
        returncode = 1
        stdout = ""
        stderr = "simulated failure"

    def fake_run(*args, **kwargs):
        return DummyProc()

    monkeypatch.setattr(env, "logger", dummy_logger)
    monkeypatch.setattr(env.subprocess, "run", fake_run)
    monkeypatch.setattr(env, "stop_container_event_listener", lambda: None)
    monkeypatch.setattr(env, "_await_cleanup_task", lambda: None)
    monkeypatch.setattr(env, "_await_reaper_task", lambda: None)
    monkeypatch.setattr(env, "cancel_cleanup_check", lambda: None)
    monkeypatch.setattr(env, "_WARMUP_TASKS", {})
    monkeypatch.setattr(env, "_DOCKER_CLIENT", None)
    monkeypatch.setattr(env, "_prune_volumes", lambda progress=None: 0)
    monkeypatch.setattr(env, "_prune_networks", lambda progress=None: 0)
    monkeypatch.setattr(env, "_release_pool_lock", lambda: None)
    monkeypatch.setattr(env, "_POOL_FILE_LOCK", types.SimpleNamespace(acquire=lambda: None))

    env._cleanup_pools()

    assert any(
        level == "error"
        and "failed to list stale sandbox containers" in msg
        and "simulated failure" in msg
        for level, msg in dummy_logger.records
    )
