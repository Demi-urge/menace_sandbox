import builtins
import importlib.util
import contextlib
import os
import sys
import types
import urllib.request
from pathlib import Path

import pytest

from dynamic_path_router import resolve_dir, resolve_path, repo_root

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

ROOT = repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

package_path = resolve_dir("sandbox_runner")
package = types.ModuleType("sandbox_runner")
package.__path__ = [str(package_path)]
sys.modules["sandbox_runner"] = package
# Provide a minimal environment stub to avoid heavy imports
env_stub = types.ModuleType("sandbox_runner.environment")
env_stub._patched_imports = contextlib.nullcontext  # type: ignore[attr-defined]
sys.modules["sandbox_runner.environment"] = env_stub
spec = importlib.util.spec_from_file_location(
    "sandbox_runner.workflow_sandbox_runner", resolve_path("workflow_sandbox_runner.py")  # path-ignore
)
wsr = importlib.util.module_from_spec(spec)
assert spec.loader
sys.modules[spec.name] = wsr
spec.loader.exec_module(wsr)
WorkflowSandboxRunner = wsr.WorkflowSandboxRunner


@pytest.fixture
def network_mock():
    class Resp:
        def read(self) -> bytes:  # pragma: no cover - simple accessor
            return b"mocked"

        def __enter__(self):  # pragma: no cover - context protocol
            return self

        def __exit__(self, *exc):  # pragma: no cover - context protocol
            return False

    return {"http://example.com": lambda url, *a, **kw: Resp()}


@pytest.fixture
def fs_mock():
    def _mock(path, mode, *a, **kw):  # pragma: no cover - pass-through
        return builtins.open(path, mode, *a, **kw)

    return {"open": _mock}


@pytest.fixture(autouse=True)
def _no_psutil(monkeypatch):
    monkeypatch.setattr(wsr, "psutil", None, raising=False)


def test_network_call_requires_mock(network_mock):
    def step():
        with urllib.request.urlopen("http://example.com") as resp:
            return resp.read().decode()

    runner = WorkflowSandboxRunner()
    metrics = runner.run([step], safe_mode=True)
    assert metrics.crash_count == 1
    assert "network access disabled" in (metrics.modules[0].exception or "")

    runner = WorkflowSandboxRunner()
    metrics = runner.run([step], safe_mode=True, network_mocks=network_mock)
    assert metrics.crash_count == 0
    assert metrics.modules[0].result == "mocked"


def test_file_write_outside_blocked():
    def step():
        Path("../outside.txt").write_text("data")

    runner = WorkflowSandboxRunner()
    metrics = runner.run([step], safe_mode=True)
    assert metrics.crash_count == 1
    assert "path escapes sandbox" in (metrics.modules[0].exception or "")


def test_cpu_memory_metrics_captured():
    holder: list[int] = []

    def step():
        holder.extend([0] * 100000)
        for _ in range(100000):
            pass
        return "done"

    runner = WorkflowSandboxRunner()
    metrics = runner.run([step], safe_mode=True)
    module = metrics.modules[0]
    assert module.cpu_delta >= 0
    assert module.memory_delta >= 0

    telemetry = runner.telemetry
    assert telemetry is not None
    assert telemetry["cpu_per_module"]["step"] >= 0
    assert telemetry["memory_per_module"]["step"] >= 0
