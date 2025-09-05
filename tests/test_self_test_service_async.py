import asyncio
import importlib.util
import importlib.machinery
import types
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
import os
os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

# stub optional dependencies
sys.modules.setdefault("cryptography", types.ModuleType("cryptography"))
sys.modules.setdefault("cryptography.hazmat", types.ModuleType("hazmat"))
sys.modules.setdefault("cryptography.hazmat.primitives", types.ModuleType("primitives"))
sys.modules.setdefault("cryptography.hazmat.primitives.asymmetric", types.ModuleType("asymmetric"))
sys.modules.setdefault("cryptography.hazmat.primitives.asymmetric.ed25519", types.ModuleType("ed25519"))
ed = sys.modules["cryptography.hazmat.primitives.asymmetric.ed25519"]
ed.Ed25519PrivateKey = types.SimpleNamespace(generate=lambda: object())
ed.Ed25519PublicKey = object
serialization = types.ModuleType("serialization")
primitives = sys.modules["cryptography.hazmat.primitives"]
primitives.serialization = serialization
sys.modules.setdefault("cryptography.hazmat.primitives.serialization", serialization)
jinja_mod = types.ModuleType("jinja2")
jinja_mod.Template = lambda *a, **k: None
sys.modules.setdefault("jinja2", jinja_mod)
sys.modules.setdefault("yaml", types.ModuleType("yaml"))
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
sys.modules.setdefault("env_config", types.SimpleNamespace(DATABASE_URL="sqlite:///"))
sys.modules.setdefault("httpx", types.ModuleType("httpx"))
sys.modules.setdefault("sqlalchemy", types.ModuleType("sqlalchemy"))
sys.modules.setdefault("sqlalchemy.engine", types.ModuleType("engine"))



def load_self_test_service():
    if 'menace' in sys.modules:
        del sys.modules['menace']
    pkg = types.ModuleType('menace')
    pkg.__path__ = [str(ROOT)]
    pkg.__spec__ = importlib.machinery.ModuleSpec('menace', loader=None, is_package=True)
    sys.modules['menace'] = pkg
    spec = importlib.util.spec_from_file_location('menace.self_test_service', ROOT / 'self_test_service.py')  # path-ignore
    mod = importlib.util.module_from_spec(spec)
    sys.modules['menace.self_test_service'] = mod
    spec.loader.exec_module(mod)
    return mod


sts = load_self_test_service()


def test_async_loop(monkeypatch):
    class DummyDB:
        def __init__(self):
            self.results = []
        def add_test_result(self, p, f):
            self.results.append((p, f))

    db = DummyDB()
    svc = sts.SelfTestService(db=db)

    async def fake_run_once():
        db.add_test_result(1, 0)

    monkeypatch.setattr(svc, '_run_once', fake_run_once)

    async def runner():
        loop = asyncio.get_running_loop()
        svc.run_continuous(interval=0.01, loop=loop)
        await asyncio.sleep(0.03)
        await svc.stop()

    asyncio.run(runner())
    assert db.results

@pytest.mark.asyncio
async def test_container_invocation(monkeypatch):
    import shutil
    import subprocess
    import os
    import json

    docker = shutil.which("docker")
    if not docker:
        pytest.skip("docker unavailable")
    try:
        subprocess.run(["docker", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        pytest.skip("docker unavailable")

    recorded = {}

    async def fake_exec(*cmd, **kwargs):
        recorded["cmd"] = cmd

        class P:
            returncode = 0
            stdout = asyncio.StreamReader()

            async def wait(self):
                self.stdout.feed_data(json.dumps({"summary": {"passed": 0, "failed": 0}}).encode())
                self.stdout.feed_eof()
                return None

        return P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    async def avail(self):
        return True

    monkeypatch.setattr(sts.SelfTestService, "_docker_available", avail)

    svc = sts.SelfTestService(use_container=True, container_image="img")
    await svc._run_once()

    assert recorded["cmd"][0] == "docker"
    assert "img" in recorded["cmd"]
    assert any(f"{os.getcwd()}:{os.getcwd()}:ro" in str(x) for x in recorded["cmd"])
    assert "pytest" in recorded["cmd"]
    assert any("--json-report-file=-" in str(x) for x in recorded["cmd"])


@pytest.mark.asyncio
async def test_container_failure_logs(monkeypatch):
    recorded = []

    async def fake_exec(*cmd, **kwargs):
        if "logs" in cmd:
            recorded.append(cmd)

            class P:
                returncode = 0

                async def communicate(self):
                    return b"log out", b""

                async def wait(self):
                    return None

            return P()

        recorded.append(cmd)

        class P:
            returncode = 1

            async def communicate(self):
                return b"out", b"err"

            async def wait(self):
                return None

        return P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    async def avail(self):
        return True

    async def dummy(*a, **k):
        return None

    monkeypatch.setattr(sts.SelfTestService, "_docker_available", avail)
    monkeypatch.setattr(sts.SelfTestService, "_remove_stale_containers", dummy)
    monkeypatch.setattr(sts.SelfTestService, "_force_remove_container", dummy)

    svc = sts.SelfTestService(use_container=True, container_image="img", container_retries=0)
    with pytest.raises(RuntimeError):
        await svc._run_once()

    assert any("logs" in r for r in recorded)
    assert "log out" in svc.results.get("logs", "")

