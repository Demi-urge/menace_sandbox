import importlib.util
import importlib.machinery
import types
import sys
import os
import asyncio
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

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
sys.modules.setdefault("jinja2", types.SimpleNamespace(Template=lambda *a, **k: None))
sys.modules.setdefault("yaml", types.ModuleType("yaml"))
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
sys.modules.setdefault("env_config", types.SimpleNamespace(DATABASE_URL="sqlite:///"))
pydantic_mod = types.ModuleType("pydantic")
pydantic_dc = types.ModuleType("dataclasses")
pydantic_dc.dataclass = lambda *a, **k: (lambda cls: cls)
pydantic_mod.dataclasses = pydantic_dc
pydantic_mod.BaseModel = object
sys.modules.setdefault("pydantic", pydantic_mod)
sys.modules.setdefault("pydantic.dataclasses", pydantic_dc)
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


def load_self_test_service():
    if "menace" in sys.modules:
        del sys.modules["menace"]
    sys.modules.setdefault('data_bot', types.SimpleNamespace(DataBot=object))
    sys.modules.setdefault('error_bot', types.SimpleNamespace(ErrorDB=object))
    sys.modules.setdefault('error_logger', types.SimpleNamespace(ErrorLogger=object))
    sys.modules.setdefault('knowledge_graph', types.SimpleNamespace(KnowledgeGraph=object))
    pkg = types.ModuleType("menace")
    pkg.__path__ = [str(ROOT)]
    pkg.__spec__ = importlib.machinery.ModuleSpec("menace", loader=None, is_package=True)
    sys.modules["menace"] = pkg
    spec = importlib.util.spec_from_file_location(
        "menace.self_test_service", ROOT / "self_test_service.py"  # path-ignore
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["menace.self_test_service"] = mod
    spec.loader.exec_module(mod)
    return mod


sts = load_self_test_service()
SelfTestService = sts.SelfTestService


class DummyBuilder:
    def refresh_db_weights(self):
        pass

    def build_context(self, *a, **k):
        return "", "", {}


def test_cleanup_lock_stress(monkeypatch):
    concurrent = 0
    max_concurrent = 0
    calls = 0

    async def fake_exec(*cmd, **kwargs):
        nonlocal concurrent, max_concurrent, calls
        if cmd[0] == "docker" and cmd[1] == "--version":
            class P:
                returncode = 0

                async def wait(self):
                    return None

            return P()
        if cmd[0] == "docker" and cmd[1] == "ps":
            calls += 1
            concurrent += 1
            max_concurrent = max(max_concurrent, concurrent)
            await asyncio.sleep(0.05)
            concurrent -= 1
            class P:
                returncode = 0

                async def communicate(self):
                    return b"", b""

                async def wait(self):
                    return None

            return P()
        if cmd[0] == "docker" and cmd[1] == "rm":
            class P:
                returncode = 0

                async def wait(self):
                    return None

            return P()
        raise RuntimeError("unexpected command")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    svc1 = SelfTestService(use_container=True, context_builder=DummyBuilder())
    svc2 = SelfTestService(use_container=True, context_builder=DummyBuilder())

    async def run():
        await asyncio.gather(
            svc1._cleanup_containers(),
            svc2._cleanup_containers(),
            svc1._cleanup_containers(),
        )

    asyncio.run(run())
    assert max_concurrent == 1
    assert calls == 3
    assert not svc1._container_lock.locked()
    assert not svc2._container_lock.locked()
