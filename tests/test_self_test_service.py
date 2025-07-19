import importlib.util
import json
import sys
import types
import os
import asyncio

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

sys.modules.setdefault("cryptography", types.ModuleType("cryptography"))
sys.modules.setdefault("cryptography.hazmat", types.ModuleType("hazmat"))
sys.modules.setdefault("cryptography.hazmat.primitives", types.ModuleType("primitives"))
sys.modules.setdefault(
    "cryptography.hazmat.primitives.asymmetric", types.ModuleType("asymmetric")
)
sys.modules.setdefault(
    "cryptography.hazmat.primitives.asymmetric.ed25519", types.ModuleType("ed25519")
)
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


ROOT = __import__('pathlib').Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "menace.self_test_service",
    ROOT / "self_test_service.py",
)
mod = importlib.util.module_from_spec(spec)
import sys
pkg = sys.modules.get("menace")
if pkg is not None:
    pkg.__path__ = [str(ROOT)]
spec.loader.exec_module(mod)
import menace.error_bot as eb
import types


def test_scheduler_start(monkeypatch):
    calls: list[None] = []

    async def fake_run_once():
        calls.append(None)

    svc = mod.SelfTestService()
    monkeypatch.setattr(svc, '_run_once', fake_run_once)

    async def runner():
        loop = asyncio.get_running_loop()
        svc.run_continuous(interval=0.01, loop=loop)
        await asyncio.sleep(0.03)
        await svc.stop()

    asyncio.run(runner())
    assert calls


def test_failure_logs_telemetry(tmp_path, monkeypatch):
    db = eb.ErrorDB(tmp_path / "e.db")
    svc = mod.SelfTestService(db)

    async def fail_exec(*cmd, **kwargs):
        path = None
        for i, a in enumerate(cmd):
            s = str(a)
            if s.startswith("--json-report-file"):
                path = s.split("=", 1)[1] if "=" in s else cmd[i + 1]
                break
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"summary": {"passed": 0, "failed": 1}}, fh)
        class P:
            returncode = 1
            async def wait(self):
                return None
        return P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fail_exec)
    asyncio.run(svc._run_once())
    cur = db.conn.execute("SELECT COUNT(*) FROM telemetry")
    assert cur.fetchone()[0] == 1
    row = db.conn.execute("SELECT passed, failed FROM test_results").fetchone()
    assert row == (0, 1)


def test_success_logs_results(tmp_path, monkeypatch):
    db = eb.ErrorDB(tmp_path / "e2.db")
    svc = mod.SelfTestService(db)

    async def succeed_exec(*cmd, **kwargs):
        path = None
        for i, a in enumerate(cmd):
            s = str(a)
            if s.startswith("--json-report-file"):
                path = s.split("=", 1)[1] if "=" in s else cmd[i + 1]
                break
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"summary": {"passed": 3, "failed": 0}}, fh)
        class P:
            returncode = 0
            async def wait(self):
                return None
        return P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", succeed_exec)
    asyncio.run(svc._run_once())
    cur = db.conn.execute("SELECT COUNT(*) FROM telemetry")
    assert cur.fetchone()[0] == 0
    row = db.conn.execute("SELECT passed, failed FROM test_results").fetchone()
    assert row == (3, 0)


def test_custom_args(monkeypatch):
    recorded = {}

    async def fake_exec(*cmd, **kwargs):
        recorded['cmd'] = cmd
        path = None
        for i, a in enumerate(cmd):
            s = str(a)
            if s.startswith("--json-report-file"):
                path = s.split("=", 1)[1] if "=" in s else cmd[i + 1]
                break
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"summary": {"passed": 0, "failed": 0}}, fh)
        class P:
            returncode = 0
            async def wait(self):
                return None
        return P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    svc = mod.SelfTestService(pytest_args="-k pattern")
    asyncio.run(svc._run_once())
    assert any("-k" in str(x) for x in recorded['cmd']) and any("pattern" in str(x) for x in recorded['cmd'])


def test_parallel_workers(monkeypatch):
    calls = []

    async def fake_exec(*cmd, **kwargs):
        calls.append(cmd)
        path = None
        for i, a in enumerate(cmd):
            s = str(a)
            if s.startswith("--json-report-file"):
                path = s.split("=", 1)[1] if "=" in s else cmd[i + 1]
                break
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"summary": {"passed": 0, "failed": 0}}, fh)
        class P:
            returncode = 0
            async def wait(self):
                return None
        return P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    svc = mod.SelfTestService(pytest_args="path1 path2", workers=2)
    asyncio.run(svc._run_once())
    assert len(calls) == 2
    for cmd in calls:
        assert "-n" in cmd and "2" in cmd

