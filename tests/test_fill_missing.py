import pytest

pytest.importorskip("requests")

import types  # noqa: E402
import sys  # noqa: E402
sys.modules.setdefault("cryptography", types.ModuleType("cryptography"))
sys.modules.setdefault("cryptography.hazmat", types.ModuleType("hazmat"))
sys.modules.setdefault("cryptography.hazmat.primitives", types.ModuleType("primitives"))
sys.modules.setdefault("cryptography.hazmat.primitives.asymmetric", types.ModuleType("asymmetric"))
sys.modules.setdefault(
    "cryptography.hazmat.primitives.asymmetric.ed25519",
    types.ModuleType("ed25519"),
)  # noqa: E501
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

import menace.task_handoff_bot as thb  # noqa: E402
import menace.implementation_optimiser_bot as iob  # noqa: E402


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
    builder = types.SimpleNamespace(refresh_db_weights=lambda *a, **k: None)
    bot = iob.ImplementationOptimiserBot(context_builder=builder)
    pkg = bot.fill_missing(_package(deps=["other"]))
    code = pkg.tasks[0].code
    assert "import other" in code
    assert "def run()" in code
    assert "Returns:" in code


class DummyEngine:
    def __init__(self, *, context_builder):
        self.context_builder = context_builder

    def generate_helper(self, desc: str) -> str:
        return "def helper():\n    pass\n"


def test_fill_missing_uses_engine():
    builder = types.SimpleNamespace(refresh_db_weights=lambda *a, **k: None)
    engine = DummyEngine(context_builder=builder)
    bot = iob.ImplementationOptimiserBot(engine=engine, context_builder=builder)
    pkg = bot.fill_missing(_package())
    code = pkg.tasks[0].code
    assert code.startswith("def helper():")
    assert "pass" in code


def test_fill_missing_shell_language():
    builder = types.SimpleNamespace(refresh_db_weights=lambda *a, **k: None)
    bot = iob.ImplementationOptimiserBot(context_builder=builder)
    meta = {"language": "bash"}
    pkg = bot.fill_missing(_package(name="sh", deps=["echo hi"], meta=meta))
    code = pkg.tasks[0].code
    assert code.startswith("#!/bin/sh")
    assert "echo hi" in code
