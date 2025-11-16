import types
import sys
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
sys.modules.setdefault("env_config", types.SimpleNamespace(DATABASE_URL="sqlite:///:memory:"))
sys.modules.setdefault("httpx", types.ModuleType("httpx"))
jinja_mod = types.ModuleType("jinja2")
jinja_mod.Template = lambda *a, **k: None
sys.modules.setdefault("jinja2", jinja_mod)
sys.modules.setdefault("yaml", types.ModuleType("yaml"))
sys.modules.setdefault("numpy", types.ModuleType("numpy"))

import menace.implementation_optimiser_bot as iob


def test_optimise_python_removes_pass_when_others_present():
    code = "def run():\n    pass\n    return 1\n"
    optimised = iob.ImplementationOptimiserBot._optimise_python(code)
    assert "pass" not in optimised


def test_optimise_python_retains_lonely_pass():
    code = "if True:\n    pass\n"
    optimised = iob.ImplementationOptimiserBot._optimise_python(code)
    assert "pass" in optimised
    compile(optimised, "<test>", "exec")


def test_optimise_python_removes_unused_imports():
    code = "import os\nimport sys\n\n" "def run():\n    return os.name\n"
    optimised = iob.ImplementationOptimiserBot._optimise_python(code)
    assert "import sys" not in optimised
    assert "import os" in optimised

