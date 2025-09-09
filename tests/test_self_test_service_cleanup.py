import asyncio
import importlib.util
import importlib.machinery
import types
import sys
import logging
from pathlib import Path
import os
import pytest
import subprocess

# replicate environment stubs similar to other tests
ROOT = Path(__file__).resolve().parents[1]

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

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
sys.modules.setdefault("jinja2", types.SimpleNamespace(Template=lambda *a, **k: None))
sys.modules.setdefault("yaml", types.ModuleType("yaml"))
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
sys.modules.setdefault("env_config", types.SimpleNamespace(DATABASE_URL="sqlite:///"))
sys.modules.setdefault("httpx", types.ModuleType("httpx"))
sys.modules.setdefault("sqlalchemy", types.ModuleType("sqlalchemy"))
sys.modules.setdefault("sqlalchemy.engine", types.ModuleType("engine"))
orphan_stub = types.ModuleType("orphan_discovery")
orphan_stub.append_orphan_cache = lambda *a, **k: None
orphan_stub.append_orphan_classifications = lambda *a, **k: None
orphan_stub.prune_orphan_cache = lambda *a, **k: None
orphan_stub.load_orphan_cache = lambda *a, **k: {}
sandbox_runner = types.ModuleType("sandbox_runner")
sandbox_runner.discover_recursive_orphans = lambda *a, **k: {}
sys.modules.setdefault("sandbox_runner", sandbox_runner)
sys.modules.setdefault("sandbox_runner.orphan_discovery", orphan_stub)
sys.modules.setdefault("orphan_discovery", orphan_stub)
pydantic_mod = types.ModuleType("pydantic")
pydantic_dc = types.ModuleType("dataclasses")
pydantic_dc.dataclass = lambda *a, **k: (lambda cls: cls)
pydantic_mod.dataclasses = pydantic_dc
pydantic_mod.BaseModel = object
sys.modules.setdefault("pydantic", pydantic_mod)
sys.modules.setdefault("pydantic.dataclasses", pydantic_dc)


def load_self_test_service():
    if 'menace' in sys.modules:
        del sys.modules['menace']
    sys.modules.setdefault('data_bot', types.SimpleNamespace(DataBot=object))
    sys.modules.setdefault('error_bot', types.SimpleNamespace(ErrorDB=object))
    sys.modules.setdefault('error_logger', types.SimpleNamespace(ErrorLogger=object))
    sys.modules.setdefault('knowledge_graph', types.SimpleNamespace(KnowledgeGraph=object))
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


class DummyBuilder:
    def refresh_db_weights(self):
        pass

    def build_context(self, *a, **k):
        return "", "", {}


def test_force_remove_container_retries_and_logs(monkeypatch, caplog):
    calls = []

    async def fake_exec(*cmd, **kwargs):
        calls.append(cmd)
        ret = 1 if len(calls) == 1 else 0

        class P:
            returncode = ret
            async def communicate(self):
                return b"", b"boom" if ret else b""
            async def wait(self):
                return None
        return P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    svc = sts.SelfTestService(
        use_container=True,
        container_retries=1,
        context_builder=DummyBuilder(),
    )
    caplog.set_level(logging.WARNING)
    asyncio.run(svc._force_remove_container("cid"))
    assert len(calls) == 2
    assert "failed to remove container" in caplog.text


def test_remove_stale_containers_retries_and_logs(monkeypatch, caplog):
    ps_calls = []

    async def fake_exec(*cmd, **kwargs):
        if "ps" in cmd:
            ps_calls.append(True)
            ret = 1 if len(ps_calls) == 1 else 0
            class P:
                returncode = ret
                async def communicate(self):
                    if ret:
                        return b"", b"err"
                    return b"abc\n", b""
                async def wait(self):
                    return None
            return P()
        raise AssertionError("unexpected call")

    removed = []

    async def dummy_force(self, cid):
        removed.append(cid)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(sts.SelfTestService, "_force_remove_container", dummy_force)
    svc = sts.SelfTestService(
        use_container=True,
        container_retries=1,
        context_builder=DummyBuilder(),
    )
    caplog.set_level(logging.WARNING)
    asyncio.run(svc._remove_stale_containers())
    assert len(ps_calls) == 2
    assert removed == ["abc"]
    assert "failed to list stale containers" in caplog.text


def _get(gauge):
    if hasattr(gauge, "_value"):
        return gauge._value.get()
    return gauge.labels().get()


def test_force_remove_failure_metric(monkeypatch):
    async def fake_exec(*cmd, **kwargs):
        class P:
            returncode = 1

            async def communicate(self):
                return b"", b"err"

            async def wait(self):
                return None

        return P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    sts.self_test_container_failures_total.set(0)
    svc = sts.SelfTestService(
        use_container=True,
        container_retries=1,
        context_builder=DummyBuilder(),
    )
    asyncio.run(svc._force_remove_container("cid"))
    assert _get(sts.self_test_container_failures_total) == 1


def test_remove_stale_containers_failure_metric(monkeypatch):
    async def fake_exec(*cmd, **kwargs):
        class P:
            returncode = 1

            async def communicate(self):
                return b"", b"err"

            async def wait(self):
                return None

        return P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    sts.self_test_container_failures_total.set(0)
    svc = sts.SelfTestService(
        use_container=True,
        container_retries=1,
        context_builder=DummyBuilder(),
    )
    asyncio.run(svc._remove_stale_containers())
    assert _get(sts.self_test_container_failures_total) == 1


def test_run_module_harness_cleans_stub(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(sts.settings, "sandbox_data_dir", tmp_path)
    mod_file = tmp_path / "m.py"  # path-ignore
    mod_file.write_text("def f():\n    return 1\n")

    def fake_run(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0], stderr=b"boom")

    monkeypatch.setattr(subprocess, "run", fake_run)
    svc = sts.SelfTestService(context_builder=DummyBuilder())
    with caplog.at_level(logging.ERROR):
        passed, _, _ = svc._run_module_harness(mod_file.as_posix())
    assert not passed
    stub_root = tmp_path / "selftest_stubs"
    assert not list(stub_root.glob("*"))
    assert any("module harness failed" in r.message for r in caplog.records)


def test_cleanup_containers_failure_metric(monkeypatch):
    async def fake_exec(*cmd, **kwargs):
        class P:
            returncode = 1

            async def communicate(self):
                return b"", b"err"

            async def wait(self):
                return None

        return P()

    async def avail(self):
        return True

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(sts.SelfTestService, "_docker_available", avail)
    sts.self_test_container_failures_total.set(0)
    svc = sts.SelfTestService(
        use_container=True,
        container_retries=1,
        context_builder=DummyBuilder(),
    )
    asyncio.run(svc._cleanup_containers())
    assert _get(sts.self_test_container_failures_total) == 1
