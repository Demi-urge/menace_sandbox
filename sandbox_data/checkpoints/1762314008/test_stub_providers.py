import sys
from pathlib import Path

import importlib.metadata as metadata
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import pytest
os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

sp = _load("stub_providers", ROOT / "sandbox_runner" / "stub_providers.py")  # path-ignore
from sandbox_settings import SandboxSettings


def test_discover_stub_providers_settings(monkeypatch):
    def prov_a(stubs, ctx):
        return stubs + [{"a": 1}]

    def prov_b(stubs, ctx):
        return stubs + [{"b": 2}]

    class EP:
        def __init__(self, name, func):
            self.name = name
            self._func = func

        def load(self):  # pragma: no cover - trivial
            return self._func

    def fake_entry_points(*, group):
        if group == sp.STUB_PROVIDER_GROUP:
            return [EP("a", prov_a), EP("b", prov_b)]
        return []

    monkeypatch.setattr(metadata, "entry_points", fake_entry_points)

    settings = SandboxSettings(stub_providers=["a"])
    providers = sp.discover_stub_providers(settings)
    assert providers == [prov_a]

    settings = SandboxSettings(disabled_stub_providers=["a"])
    providers = sp.discover_stub_providers(settings)
    assert providers == [prov_b]


def test_invalid_and_failing_providers(monkeypatch):
    def good(stubs, ctx):
        return stubs

    def bad(stubs):  # missing ctx argument
        return stubs

    class GoodEP:
        name = "good"

        def load(self):  # pragma: no cover - trivial
            return good

    class BadEP:
        name = "bad"

        def load(self):  # pragma: no cover - trivial
            return bad

    class FailEP:
        name = "fail"

        def load(self):  # pragma: no cover - trivial
            raise RuntimeError("boom")

    def fake_entry_points(*, group):
        if group == sp.STUB_PROVIDER_GROUP:
            return [GoodEP(), BadEP(), FailEP()]
        return []

    monkeypatch.setattr(metadata, "entry_points", fake_entry_points)
    with pytest.raises(RuntimeError):
        sp.discover_stub_providers()
