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

import importlib.util  # noqa: E402

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "menace", root / "__init__.py", submodule_search_locations=[str(root)]  # path-ignore
)
pkg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pkg)
sys.modules["menace"] = pkg
sys.modules.setdefault("menace.self_coding_engine", types.SimpleNamespace(SelfCodingEngine=object))
import menace.automated_debugger as ad  # noqa: E402
from menace.vector_service.context_builder import ContextBuilder  # noqa: E402


class DummyEngine:
    def __init__(self):
        self.called = False

    def apply_patch(self, path: Path, description: str, *, target_region=None, **kw):
        self.called = True


class DummyTelem:
    def __init__(self, log: str = ""):
        self.log = log

    def recent_errors(self, limit: int = 5):
        return [self.log]


class DummyBuilder(ContextBuilder):
    def __init__(self):
        pass

    def build_context(self, query: str, **kwargs):
        return {}

    def refresh_db_weights(self):
        pass


class DummyManager:
    def __init__(self, engine=None):
        self.engine = engine
        self.called = False

    def run_patch(self, path, mode, **kwargs):
        self.called = True
        if self.engine is not None:
            self.engine.apply_patch(path, mode, **kwargs)


def test_analyse_and_fix(monkeypatch, tmp_path):
    mod = tmp_path / "mod.py"  # path-ignore
    mod.write_text("def x():\n    pass\n")
    log = f"File '{mod}', line 1, in x"
    eng = DummyEngine()
    manager = DummyManager(eng)
    dbg = ad.AutomatedDebugger(
        DummyTelem(log), eng, DummyBuilder(), manager=manager
    )
    monkeypatch.setitem(
        sys.modules,
        "sandbox_runner",
        types.SimpleNamespace(integrate_new_orphans=lambda p: None),
    )
    monkeypatch.setattr(
        ad.tempfile,
        "NamedTemporaryFile",
        lambda *a, **k: open(tmp_path / "t.py", "w+"),
    )  # path-ignore
    monkeypatch.setattr(
        ad.subprocess,
        "run",
        lambda *a, **k: types.SimpleNamespace(returncode=0, stderr=b"", stdout=""),
    )
    dbg.analyse_and_fix()
    assert eng.called


def test_generate_tests_absolute_path(monkeypatch, tmp_path):
    eng = DummyEngine()
    dbg = ad.AutomatedDebugger(
        DummyTelem(), eng, DummyBuilder(), manager=DummyManager(eng)
    )
    mod = tmp_path / "mod.py"  # path-ignore
    mod.write_text("def x():\n    pass\n")
    monkeypatch.chdir(tmp_path)
    log = f"File '{mod}', line 1, in x"
    tests = dbg._generate_tests([log])
    assert "import_module('mod')" in tests[0]


def test_generate_tests_traceback(monkeypatch, tmp_path):
    eng = DummyEngine()
    dbg = ad.AutomatedDebugger(
        DummyTelem(), eng, DummyBuilder(), manager=DummyManager(eng)
    )
    a = tmp_path / "a.py"  # path-ignore
    b = tmp_path / "b.py"  # path-ignore
    a.write_text("def foo():\n    bar()\n")
    b.write_text("def bar():\n    pass\n")
    monkeypatch.chdir(tmp_path)
    log = (
        "Traceback (most recent call last):\n"
        f"  File '{a}', line 1, in foo\n"
        f"  File '{b}', line 1, in bar\n"
        "ValueError: boom\n"
    )
    tests = dbg._generate_tests([log])
    assert "import_module('b')" in tests[0]
    assert "getattr(mod, 'bar'" in tests[0]
