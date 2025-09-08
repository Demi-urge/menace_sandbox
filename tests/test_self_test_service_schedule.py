import importlib.machinery
import types
import sys
from pathlib import Path
import asyncio
import pytest

ROOT = Path(__file__).resolve().parents[1]

# mimic environment stubs like other tests
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


def load_self_test_service():
    if 'menace' in sys.modules:
        del sys.modules['menace']
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


def test_run_scheduled(monkeypatch, tmp_path):
    class DummyBuilder:
        def refresh_db_weights(self):
            pass

        def build_context(self, *a, **k):
            if k.get('return_metadata'):
                return '', {}
            return ''

    svc = sts.SelfTestService(history_path=tmp_path / 'h.json', use_container=True, context_builder=DummyBuilder())
    calls = []

    async def fake_run_once():
        calls.append(True)

    monkeypatch.setattr(svc, '_run_once', fake_run_once)
    monkeypatch.setattr(sts.time, 'sleep', lambda s: None)

    svc.run_scheduled(interval=0.01, runs=3)

    assert len(calls) == 3
