import importlib
import sys
import types

import pytest

from tests.test_menace_master import _setup_mm_stubs


class _DummyBroker:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def advertise(self, **kwargs: object) -> None:
        self.calls.append(kwargs)


def _stub_coding_bot_interface(monkeypatch: pytest.MonkeyPatch) -> tuple[types.ModuleType, _DummyBroker, list[tuple]]:
    broker = _DummyBroker()
    recorded: list[tuple] = []

    def advertise_bootstrap_placeholder(*, dependency_broker=None, pipeline=None, manager=None, owner=True):
        recorded.append((pipeline, manager, owner))
        sentinel = manager or types.SimpleNamespace()
        setattr(sentinel, "_self_coding_bootstrap_placeholder", True)
        pipeline_candidate = pipeline or types.SimpleNamespace(manager=sentinel)
        setattr(pipeline_candidate, "_self_coding_bootstrap_placeholder", True)
        broker.advertise(pipeline=pipeline_candidate, sentinel=sentinel, owner=owner)
        return pipeline_candidate, sentinel

    def _is_bootstrap_placeholder(candidate):
        return bool(getattr(candidate, "_self_coding_bootstrap_placeholder", False))

    stub = types.ModuleType("coding_bot_interface")
    stub.advertise_bootstrap_placeholder = advertise_bootstrap_placeholder
    stub._bootstrap_dependency_broker = lambda: broker
    stub._is_bootstrap_placeholder = _is_bootstrap_placeholder

    monkeypatch.setitem(sys.modules, "coding_bot_interface", stub)
    monkeypatch.setitem(sys.modules, "menace.coding_bot_interface", stub)
    monkeypatch.setitem(sys.modules, "menace_sandbox.coding_bot_interface", stub)

    return stub, broker, recorded


def test_manual_bootstrap_advertises_placeholder(monkeypatch):
    stub, broker, recorded = _stub_coding_bot_interface(monkeypatch)

    menace_pkg = types.ModuleType("menace")
    menace_pkg.RAISE_ERRORS = True
    monkeypatch.setitem(sys.modules, "menace", menace_pkg)

    monkeypatch.setitem(
        sys.modules,
        "menace_sandbox.environment_bootstrap",
        types.SimpleNamespace(EnvironmentBootstrapper=type("EnvironmentBootstrapper", (), {})),
    )
    monkeypatch.setitem(
        sys.modules,
        "menace_sandbox.sandbox_runner.bootstrap",
        types.SimpleNamespace(
            bootstrap_environment=lambda *_, **__: None,
            ensure_autonomous_launch=lambda *_, **__: None,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "menace_sandbox.sandbox_settings",
        types.SimpleNamespace(SandboxSettings=type("SandboxSettings", (), {})),
    )

    module = importlib.reload(importlib.import_module("manual_bootstrap"))

    assert recorded, "placeholder helper was not invoked"
    assert broker.calls, "dependency broker did not capture placeholder advertisement"
    advertised = broker.calls[-1]
    assert stub._is_bootstrap_placeholder(advertised.get("pipeline"))
    assert stub._is_bootstrap_placeholder(advertised.get("sentinel"))


def test_menace_master_advertises_placeholder(monkeypatch):
    stub, broker, recorded = _stub_coding_bot_interface(monkeypatch)
    _setup_mm_stubs(monkeypatch)

    module = importlib.reload(importlib.import_module("menace_master"))

    assert recorded, "placeholder helper was not invoked during menace_master import"
    assert broker.calls, "dependency broker did not capture menace_master placeholder"
    advertised = broker.calls[-1]
    assert stub._is_bootstrap_placeholder(advertised.get("pipeline"))
    assert stub._is_bootstrap_placeholder(advertised.get("sentinel"))
    assert getattr(module, "_BOOTSTRAP_PLACEHOLDER", None) is advertised.get("pipeline")
