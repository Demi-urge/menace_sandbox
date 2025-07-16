import sys
import types
from pathlib import Path

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

jinja = types.ModuleType("jinja2")
jinja.Template = lambda *a, **k: None
sys.modules.setdefault("jinja2", jinja)
yaml = types.ModuleType("yaml")
sys.modules.setdefault("yaml", yaml)
sys.modules.setdefault("numpy", types.ModuleType("numpy"))

import menace.automated_debugger as ad


class DummyEngine:
    def __init__(self):
        self.called = False

    def apply_patch(self, path: Path, description: str, **kw):
        self.called = True


class DummyTelem:
    def recent_errors(self, limit: int = 5):
        return ["error"]


def test_analyse_and_fix(monkeypatch, tmp_path):
    eng = DummyEngine()
    dbg = ad.AutomatedDebugger(DummyTelem(), eng)
    monkeypatch.setattr(ad.tempfile, "NamedTemporaryFile", lambda *a, **k: open(tmp_path / "t.py", "w+"))
    monkeypatch.setattr(ad.subprocess, "run", lambda *a, **k: types.SimpleNamespace(returncode=0, stderr=b""))
    dbg.analyse_and_fix()
    assert eng.called


def test_generate_tests_absolute_path(monkeypatch, tmp_path):
    eng = DummyEngine()
    dbg = ad.AutomatedDebugger(DummyTelem(), eng)
    mod = tmp_path / "mod.py"
    mod.write_text("def x():\n    pass\n")
    monkeypatch.chdir(tmp_path)
    log = f"File '{mod}', line 1, in x"
    tests = dbg._generate_tests([log])
    assert "import_module('mod')" in tests[0]
