import importlib.util
import importlib.machinery
import os
import runpy
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

# Stub optional dependencies expected by self_test_service
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
jinja_mod = types.ModuleType("jinja2")
jinja_mod.Template = lambda *a, **k: None
sys.modules.setdefault("jinja2", jinja_mod)
sys.modules.setdefault("yaml", types.ModuleType("yaml"))
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
sys.modules.setdefault("env_config", types.SimpleNamespace(DATABASE_URL="sqlite:///"))
sys.modules.setdefault("httpx", types.ModuleType("httpx"))
sys.modules.setdefault("sqlalchemy", types.ModuleType("sqlalchemy"))
sys.modules.setdefault("sqlalchemy.engine", types.ModuleType("engine"))
sys.modules.setdefault("data_bot", types.SimpleNamespace(DataBot=object, MetricsDB=object))
sys.modules.setdefault("menace.data_bot", sys.modules["data_bot"])
sys.modules.setdefault("error_bot", types.SimpleNamespace(ErrorDB=object))
sys.modules.setdefault("menace.error_bot", sys.modules["error_bot"])
sys.modules.setdefault(
    "error_logger", types.SimpleNamespace(ErrorLogger=lambda *a, **k: None)
)
sys.modules.setdefault("menace.error_logger", sys.modules["error_logger"])
sys.modules.setdefault("sandbox_runner", types.ModuleType("sandbox_runner"))
od = types.SimpleNamespace(
    append_orphan_cache=lambda *a, **k: None,
    append_orphan_classifications=lambda *a, **k: None,
    prune_orphan_cache=lambda *a, **k: None,
    load_orphan_cache=lambda *a, **k: {},
)
sys.modules.setdefault("sandbox_runner.orphan_discovery", od)
sys.modules.setdefault("orphan_discovery", od)


class _Gauge:
    def __init__(self, *a, **k):
        self._value = 0

    def set(self, v):
        self._value = v

    def labels(self, *a, **k):
        return self

    def get(self):
        return self._value


me = types.SimpleNamespace(
    Gauge=_Gauge,
    orphan_modules_reintroduced_total=_Gauge(),
    orphan_modules_passed_total=_Gauge(),
    orphan_modules_tested_total=_Gauge(),
    orphan_modules_failed_total=_Gauge(),
    orphan_modules_reclassified_total=_Gauge(),
    orphan_modules_redundant_total=_Gauge(),
    orphan_modules_legacy_total=_Gauge(),
    error_bot_exceptions=None,
)

sys.modules.setdefault("metrics_exporter", me)
sys.modules.setdefault("menace.metrics_exporter", me)


def load_self_test_service():
    if "menace" in sys.modules:
        del sys.modules["menace"]
    pkg = types.ModuleType("menace")
    pkg.__path__ = [str(ROOT)]
    pkg.__spec__ = importlib.machinery.ModuleSpec("menace", loader=None, is_package=True)
    pkg.RAISE_ERRORS = False
    sys.modules["menace"] = pkg
    spec = importlib.util.spec_from_file_location(
        "menace.self_test_service", ROOT / "self_test_service.py"  # path-ignore
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["menace.self_test_service"] = mod
    spec.loader.exec_module(mod)
    return mod


sts = load_self_test_service()


def test_stub_supplies_dummy_args(tmp_path, monkeypatch):
    mod_path = tmp_path / "sample_mod.py"  # path-ignore
    mod_path.write_text(
        "called = []\n"
        "def func(a, b):\n"
        "    called.append((a, b))\n"
        "class C:\n"
        "    def __init__(self, x, y):\n"
        "        called.append((x, y))\n",
        encoding="utf-8",
    )

    sys.path.insert(0, str(tmp_path))
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))

    class DummyBuilder:
        def refresh_db_weights(self):
            pass

        def build_context(self, *a, **k):
            if k.get("return_metadata"):
                return "", {}
            return ""

    svc = sts.SelfTestService(context_builder=DummyBuilder())
    stub_path = svc._generate_pytest_stub(str(mod_path))

    ns = runpy.run_path(stub_path)
    ns["test_stub"]()

    mod = sys.modules["sample_mod"]
    assert len(mod.called) == 2
    assert all(len(args) == 2 for args in mod.called)

    stub_dir = stub_path.parent
    stub_path.unlink()
    stub_dir.rmdir()
    stub_dir.parent.rmdir()
    stub_dir.parent.parent.rmdir()
    sys.path.remove(str(tmp_path))
    sys.modules.pop("sample_mod", None)

