import pytest

pytest.importorskip("requests")

import types
import sys
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
sys.modules.setdefault("env_config", types.SimpleNamespace(DATABASE_URL="sqlite:///:memory:"))
sys.modules.setdefault("httpx", types.ModuleType("httpx"))
jinja_mod = types.ModuleType("jinja2")
jinja_mod.Template = lambda *a, **k: None
sys.modules.setdefault("jinja2", jinja_mod)
sys.modules.setdefault("yaml", types.ModuleType("yaml"))
sys.modules.setdefault("numpy", types.ModuleType("numpy"))

import menace.task_handoff_bot as thb
import menace.implementation_optimiser_bot as iob


def _package(name="t", deps=None, *, meta=None):
    return thb.TaskPackage(
        tasks=[
            thb.TaskInfo(
                name=name,
                dependencies=deps or [],
                resources={},
                schedule="once",
                code="",
                metadata=meta or {},
            )
        ]
    )


def test_fill_missing_provides_default():
    bot = iob.ImplementationOptimiserBot()
    pkg = bot.fill_missing(_package(deps=["other"]))
    code = pkg.tasks[0].code
    assert "import other" in code
    assert "def run()" in code
    assert "Returns:" in code


class DummyEngine:
    def generate_helper(self, desc: str) -> str:
        return "def helper():\n    pass\n"


def test_fill_missing_uses_engine():
    bot = iob.ImplementationOptimiserBot(engine=DummyEngine())
    pkg = bot.fill_missing(_package())
    code = pkg.tasks[0].code
    assert code.startswith("def helper():")
    assert "pass" in code


def test_fill_missing_shell_language():
    bot = iob.ImplementationOptimiserBot()
    meta = {"language": "bash"}
    pkg = bot.fill_missing(_package(name="sh", deps=["echo hi"], meta=meta))
    code = pkg.tasks[0].code
    assert code.startswith("#!/bin/sh")
    assert "echo hi" in code
