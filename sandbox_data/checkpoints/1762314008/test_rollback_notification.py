import sys
import types

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

import types as _types
import menace.rollback_manager as rm


class DummyClient:
    def __init__(self, fail_for=None):
        self.fail_for = set(fail_for or [])
        self.calls = []

    def rollback(self, node: str, patch_id: str) -> bool:
        self.calls.append((node, patch_id))
        return node not in self.fail_for


def test_rollback_notify_rpc(tmp_path):
    mgr = rm.RollbackManager(str(tmp_path / "rb.db"))
    for n in ("n1", "n2"):
        mgr.register_patch("p1", n)
    client = DummyClient()
    mgr.rollback("p1", rpc_client=client)
    assert not mgr.applied_patches()
    assert set(client.calls) == {("n1", "p1"), ("n2", "p1")}


def test_rollback_notify_http_failure(tmp_path, monkeypatch):
    mgr = rm.RollbackManager(str(tmp_path / "rb2.db"))
    for n in ("n1", "n2"):
        mgr.register_patch("p2", n)

    calls = []

    class Resp:
        def __init__(self, code):
            self.status_code = code

    def fake_post(url, json=None, timeout=5):
        calls.append(url)
        return Resp(500)

    monkeypatch.setattr(rm, "requests", _types.SimpleNamespace(post=fake_post))
    endpoints = {"n1": "http://n1", "n2": "http://n2"}
    mgr.rollback("p2", endpoints=endpoints)
    assert not mgr.applied_patches()
    assert calls == ["http://n1/rollback", "http://n2/rollback"]
