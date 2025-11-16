import sys
import types

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

jinja = types.ModuleType("jinja2")
jinja.Template = lambda *a, **k: None
sys.modules.setdefault("jinja2", jinja)
yaml = types.ModuleType("yaml")
sys.modules.setdefault("yaml", yaml)
sys.modules.setdefault("numpy", types.ModuleType("numpy"))

import menace.advanced_error_management as aem


class DummyDB:
    def fetch(self, limit: int = 10):
        class Col(list):
            def mean(self):
                return sum(self) / len(self) if self else 0.0

        class DF:
            def __init__(self):
                self.rows = [{"cpu": 90.0}]

            @property
            def empty(self):
                return False

            def __getitem__(self, key):
                return Col([r[key] for r in self.rows])

        return DF()


class Resp:
    status_code = 200

    @staticmethod
    def json():
        return {"status": "ok"}


def test_autoscale_called(monkeypatch):
    calls = []

    def post(url, json=None, timeout=2):
        calls.append(url)
        return Resp()

    monkeypatch.setattr(aem, "requests", types.SimpleNamespace(post=post))
    alloc = aem.PredictiveResourceAllocator(DummyDB(), autoscale_url="http://api")
    alloc.forecast_and_allocate()
    assert calls and calls[0] == "http://api"


def test_retry_on_failure(monkeypatch):
    attempts = []

    def post(url, json=None, timeout=2):
        attempts.append(True)
        if len(attempts) < 3:
            raise RuntimeError("boom")
        return Resp()

    monkeypatch.setattr(aem, "requests", types.SimpleNamespace(post=post))
    alloc = aem.PredictiveResourceAllocator(DummyDB(), autoscale_url="http://api")
    alloc.forecast_and_allocate()
    assert len(attempts) == 3
