import types

import menace.advanced_error_management as aem


class DummyClient:
    def __init__(self, success_for):
        self.success_for = set(success_for)
        self.calls = []

    def rollback(self, node: str, patch_id: str) -> bool:  # pragma: no cover - simple stub
        self.calls.append((node, patch_id))
        return node in self.success_for


def test_quorum_success(tmp_path):
    mgr = aem.AutomatedRollbackManager(str(tmp_path / "rb.db"))
    for n in ("n1", "n2", "n3"):
        mgr.register_patch("p1", n)

    client = DummyClient({"n1", "n2", "n3"})
    ok = mgr.auto_rollback("p1", ["n1", "n2", "n3"], rpc_client=client)

    assert ok
    assert not mgr.applied_patches()


def test_quorum_failure(tmp_path, monkeypatch):
    mgr = aem.AutomatedRollbackManager(str(tmp_path / "rb.db"))
    for n in ("n1", "n2", "n3", "n4"):
        mgr.register_patch("p2", n)

    calls = []

    class Resp:
        def __init__(self, code):
            self.status_code = code

    def fake_post(url, json=None, timeout=5):  # pragma: no cover - simple stub
        calls.append(url)
        # Only first request succeeds
        code = 200 if len(calls) == 1 else 500
        return Resp(code)

    monkeypatch.setattr(aem, "requests", types.SimpleNamespace(post=fake_post))

    endpoints = {n: f"http://{n}" for n in ("n1", "n2", "n3", "n4")}
    ok = mgr.auto_rollback("p2", list(endpoints), endpoints=endpoints)

    assert not ok
    # Patch should remain because quorum not reached
    assert mgr.applied_patches()

