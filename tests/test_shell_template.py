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

import menace.implementation_optimiser_bot as iob
import menace.task_handoff_bot as thb


def test_shell_template_no_deps():
    t = thb.TaskInfo(name="a", dependencies=[], resources={}, schedule="once", code="", metadata={})
    code = iob.ImplementationOptimiserBot._shell_template(t, "desc")
    assert "set -e" in code
    assert "echo 'desc'" in code
    assert "exit 0" in code
    assert "exit 1" in code


def test_shell_template_with_deps():
    t = thb.TaskInfo(name="b", dependencies=["echo hi"], resources={}, schedule="once", code="", metadata={})
    code = iob.ImplementationOptimiserBot._shell_template(t, "desc")
    assert "echo hi" in code
    assert "exit 0" in code
    assert "exit 1" in code
