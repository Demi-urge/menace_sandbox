import types
import importlib.util

from dynamic_path_router import resolve_path

ROOT = __import__('pathlib').Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "menace.vault_secret_provider",
    resolve_path("vault_secret_provider.py"),  # path-ignore
    submodule_search_locations=[str(ROOT)],
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class DummyMgr:
    def __init__(self):
        self.tokens = {"A": "foo"}

    def get(self, name, rotate=True):
        return self.tokens.get(name, "")

    def set(self, name, token):
        self.tokens[name] = token


def test_local_fallback(monkeypatch):
    mgr = DummyMgr()
    p = mod.VaultSecretProvider(url=None, manager=mgr)
    assert p.get("A") == "foo"



def test_remote_fetch(monkeypatch):
    class DummyResp:
        status_code = 200
        text = "bar"

    class DummySess:
        def get(self, url, timeout=5):
            return DummyResp()

    monkeypatch.setitem(mod.__dict__, 'requests', types.SimpleNamespace(Session=lambda: DummySess()))
    mgr = DummyMgr()
    p = mod.VaultSecretProvider(url="http://x", manager=mgr)
    assert p.get("B") == "bar"
    assert mgr.tokens["B"] == "bar"
