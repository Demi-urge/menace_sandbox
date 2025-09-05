import asyncio
import sys
import types
from contextlib import asynccontextmanager
from pathlib import Path
import importlib.util
import atexit


sys.modules["dynamic_path_router"] = types.SimpleNamespace(
    resolve_path=lambda p: p,
    repo_root=lambda: ".",
    path_for_prompt=lambda name: name,
)
sys.modules.setdefault(
    "metrics_exporter",
    types.SimpleNamespace(
        sandbox_crashes_total=types.SimpleNamespace(
            labels=lambda *a, **k: types.SimpleNamespace(inc=lambda *a, **k: None)
        )
    ),
)
sys.modules.setdefault(
    "alert_dispatcher", types.SimpleNamespace(dispatch_alert=lambda *a, **k: None)
)

class _SandboxSettings:
    menace_light_imports = []


sys.modules.setdefault(
    "sandbox_settings", types.SimpleNamespace(SandboxSettings=_SandboxSettings)
)
baseline_module = types.SimpleNamespace(
    BaselineTracker=type("BaselineTracker", (), {}),
    TRACKER=types.SimpleNamespace(_history={}, window=1),
)
sys.modules.setdefault("self_improvement", types.ModuleType("self_improvement"))
sys.modules["self_improvement.baseline_tracker"] = baseline_module
sys.modules["self_improvement"].baseline_tracker = baseline_module
sys.modules.setdefault(
    "error_logger",
    types.SimpleNamespace(
        ErrorLogger=type("ErrorLogger", (), {"__init__": lambda self, *a, **k: None})
    ),
)

root = Path(__file__).resolve().parents[1]
package = types.ModuleType("sandbox_runner")
package.__path__ = [str(root / "sandbox_runner")]
sys.modules["sandbox_runner"] = package
sys.path.append(str(root))
env_path = root / "sandbox_runner" / "environment.py"  # path-ignore
spec = importlib.util.spec_from_file_location(
    "sandbox_runner.environment", env_path
)
environment = importlib.util.module_from_spec(spec)
sys.modules["sandbox_runner.environment"] = environment
assert spec.loader is not None
orig_register = atexit.register
atexit.register = lambda *a, **k: None
spec.loader.exec_module(environment)
atexit.register = orig_register


def test_container_snippet_path_env(monkeypatch, tmp_path):
    commands: list[list[str]] = []

    class StubContainer:
        id = "cid"

        def wait(self, timeout=None):
            return {"StatusCode": 0}

        def logs(self, stdout=True, stderr=False):  # pragma: no cover - basic stub
            return b""

        def stats(self, stream=False):  # pragma: no cover - basic stub
            return {}

        def remove(self):  # pragma: no cover - basic stub
            return None

        def exec_run(self, cmd, environment=None, workdir=None, demux=False, timeout=None):
            commands.append(cmd)

            class Result:
                exit_code = 0
                output = (b"", b"")

            return Result()

    class StubClient:
        def __init__(self):
            self.containers = self

        def run(self, image, cmd, **kwargs):
            commands.append(cmd)
            return StubContainer()

    stub_client = StubClient()

    docker_module = types.SimpleNamespace(
        errors=types.SimpleNamespace(DockerException=Exception, APIError=Exception),
        from_env=lambda: stub_client,
    )
    monkeypatch.setitem(sys.modules, "docker", docker_module)
    monkeypatch.setitem(sys.modules, "docker.errors", docker_module.errors)
    monkeypatch.setattr(environment, "_DOCKER_CLIENT", stub_client, raising=False)
    monkeypatch.setattr(environment, "_register_container_finalizer", lambda c: None)
    monkeypatch.setattr(environment, "_record_active_container", lambda cid: None)
    monkeypatch.setattr(environment, "_remove_active_container", lambda cid: None)

    @asynccontextmanager
    async def fake_pooled(image):
        yield StubContainer(), str(tmp_path)

    monkeypatch.setattr(environment, "pooled_container", fake_pooled)

    custom_path = "/alt/path.py"  # path-ignore

    env = {"CONTAINER_SNIPPET_PATH": custom_path, "CPU_LIMIT": "1"}
    asyncio.run(environment._execute_in_container("print()", env))
    assert commands.pop(0) == ["python", custom_path]

    env = {"CONTAINER_SNIPPET_PATH": custom_path}
    asyncio.run(environment._execute_in_container("print()", env))
    assert commands.pop(0) == ["python", custom_path]

