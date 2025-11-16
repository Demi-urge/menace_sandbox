import os, sys, types
from pathlib import Path
from dynamic_path_router import resolve_path
os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
sys.modules.setdefault("jinja2", types.ModuleType("jinja2"))
sys.modules["jinja2"].Template = lambda *a, **k: None
sys.modules.setdefault("cryptography", types.ModuleType("cryptography"))
sys.modules.setdefault("yaml", types.ModuleType("yaml"))
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
sandbox_runner = types.ModuleType("sandbox_runner")
sandbox_runner.discover_recursive_orphans = lambda *a, **k: {}
sys.modules.setdefault("sandbox_runner", sandbox_runner)
orphan_stub = types.ModuleType("orphan_discovery")
orphan_stub.append_orphan_cache = lambda *a, **k: None
orphan_stub.append_orphan_classifications = lambda *a, **k: None
orphan_stub.prune_orphan_cache = lambda *a, **k: None
orphan_stub.load_orphan_cache = lambda *a, **k: {}
sys.modules.setdefault("sandbox_runner.orphan_discovery", orphan_stub)
sys.modules.setdefault("orphan_discovery", orphan_stub)
import asyncio
import json
import logging
import pytest
import importlib.util
import importlib.machinery


def load_self_test_service():
    if 'menace' in sys.modules:
        del sys.modules['menace']
    sys.modules.setdefault('data_bot', types.SimpleNamespace(DataBot=object))
    sys.modules.setdefault('error_bot', types.SimpleNamespace(ErrorDB=object))
    sys.modules.setdefault('error_logger', types.SimpleNamespace(ErrorLogger=object))
    sys.modules.setdefault('knowledge_graph', types.SimpleNamespace(KnowledgeGraph=object))
    pkg = types.ModuleType('menace')
    pkg.__path__ = [str(Path(__file__).resolve().parents[1])]
    pkg.__spec__ = importlib.machinery.ModuleSpec('menace', loader=None, is_package=True)
    sys.modules['menace'] = pkg
    path = resolve_path('self_test_service.py')  # path-ignore
    spec = importlib.util.spec_from_file_location(
        'menace.self_test_service', str(path)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules['menace.self_test_service'] = mod
    spec.loader.exec_module(mod)
    return mod

sts = load_self_test_service()
SelfTestService = sts.SelfTestService


class DummyBuilder:
    def refresh_db_weights(self):
        pass

    def build_context(self, *a, **k):
        if k.get("return_metadata"):
            return "", {}
        return ""


class DummyLogger:
    def __init__(self, db=None, knowledge_graph=None):
        self.db = types.SimpleNamespace(add_test_result=lambda *a, **k: None)

    def log(self, *a, **k):
        pass


# ---------------------------------------------------------------------------

def _parse_workers(cmd):
    for i, a in enumerate(cmd):
        if a == "-n" and i + 1 < len(cmd):
            try:
                return int(cmd[i + 1])
            except ValueError:
                return 1
    return 1


# ---------------------------------------------------------------------------

def test_container_worker_distribution(monkeypatch):
    calls = []

    async def fake_exec(*cmd, **kwargs):
        calls.append(cmd)

        class P:
            returncode = 0
            stdout = asyncio.StreamReader()

            async def wait(self):
                self.stdout.feed_data(json.dumps({"summary": {"passed": 0, "failed": 0}}).encode())
                self.stdout.feed_eof()
                return None

            async def communicate(self):
                return json.dumps({"summary": {"passed": 0, "failed": 0}}).encode(), b""
        return P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(sts, "ErrorLogger", DummyLogger)

    async def avail(self):
        return True

    monkeypatch.setattr(SelfTestService, "_docker_available", avail)

    svc = SelfTestService(pytest_args="a b c", workers=5, use_container=True, container_image="img", context_builder=DummyBuilder())
    svc.run_once()

    assert not svc._container_lock.locked()
    assert len(calls) == 3
    workers = [_parse_workers(c) for c in calls]
    assert workers == [2, 2, 1]


# ---------------------------------------------------------------------------

def test_lock_released_on_docker_error(monkeypatch, caplog):
    async def fail_avail(self):
        raise RuntimeError("boom")

    monkeypatch.setattr(SelfTestService, "_docker_available", fail_avail)
    monkeypatch.setattr(sts, "ErrorLogger", DummyLogger)

    svc = SelfTestService(use_container=True, context_builder=DummyBuilder())
    caplog.set_level(logging.ERROR)
    svc.run_once()
    assert "self test run failed" in caplog.text

    assert not svc._container_lock.locked()


# ---------------------------------------------------------------------------

def test_lock_released_on_test_failure(monkeypatch, caplog):
    async def avail(self):
        return True

    monkeypatch.setattr(SelfTestService, "_docker_available", avail)
    monkeypatch.setattr(sts, "ErrorLogger", DummyLogger)

    async def fail_exec(*cmd, **kwargs):
        class P:
            returncode = 1
            stdout = asyncio.StreamReader()

            async def communicate(self):
                return json.dumps({"summary": {"passed": 0, "failed": 1}}).encode(), b""

            async def wait(self):
                return None

        return P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fail_exec)

    svc = SelfTestService(use_container=True, container_image="img", context_builder=DummyBuilder())
    caplog.set_level(logging.ERROR)
    svc.run_once()
    assert not svc._container_lock.locked()
