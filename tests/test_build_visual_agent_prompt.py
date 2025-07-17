import os
import sys
import types

# stub heavy optional deps
sys.modules.setdefault("jinja2", types.ModuleType("jinja2"))
sys.modules.setdefault("yaml", types.ModuleType("yaml"))
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
sys.modules.setdefault("cryptography", types.ModuleType("cryptography"))
sys.modules.setdefault("cryptography.hazmat", types.ModuleType("hazmat"))
sys.modules.setdefault("cryptography.hazmat.primitives", types.ModuleType("primitives"))
sys.modules.setdefault("cryptography.hazmat.primitives.asymmetric", types.ModuleType("asymmetric"))
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
sys.modules.setdefault("env_config", types.SimpleNamespace(DATABASE_URL="sqlite:///:memory:"))
sys.modules.setdefault("httpx", types.ModuleType("httpx"))
os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

import menace.self_coding_engine as sce  # noqa: E402


def test_build_visual_agent_prompt_basic():
    prompt = sce.SelfCodingEngine(None, None).build_visual_agent_prompt(
        "helper.py", "print hello", "def hello():\n    pass"
    )
    assert "### Introduction" in prompt
    assert "helper.py" in prompt
    assert "print hello" in prompt
    assert "def hello()" in prompt
