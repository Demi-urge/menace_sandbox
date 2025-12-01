import importlib
import sys
import threading
import time
import types
from types import SimpleNamespace
from typing import Any

import pytest


def _build_stub_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__file__ = __file__
    return module


class _StubBroker:
    def __init__(self) -> None:
        self.pipeline: Any | None = None
        self.sentinel: Any | None = None
        self.active_owner = False
        self.calls: list[tuple[Any | None, Any | None, bool | None]] = []

    def resolve(self) -> tuple[Any | None, Any | None]:
        return self.pipeline, self.sentinel

    def advertise(
        self,
        *,
        pipeline: Any | None = None,
        sentinel: Any | None = None,
        owner: bool | None = None,
    ) -> tuple[Any | None, Any | None]:
        if pipeline is not None:
            self.pipeline = pipeline
        if sentinel is not None:
            self.sentinel = sentinel
        if owner is not None:
            self.active_owner = owner
        self.calls.append((self.pipeline, self.sentinel, owner))
        return self.pipeline, self.sentinel

    def clear(self) -> None:  # pragma: no cover - defensive hook parity
        self.pipeline = None
        self.sentinel = None
        self.active_owner = False


def test_cascading_bootstrap_guard(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    sys.modules.setdefault("stripe", types.ModuleType("stripe"))
    import coding_bot_interface as cbi

    caplog.set_level("INFO")
    cbi._GLOBAL_BOOTSTRAP_COORDINATOR.reset()
    cbi._BOOTSTRAP_DEPENDENCY_BROKER.set(None)
    cbi._BOOTSTRAP_STATE.depth = 0
    cbi._BOOTSTRAP_STATE.helper_promotion_callbacks = []
    cbi._BOOTSTRAP_STATE.owner_promises = {}
    cbi._BOOTSTRAP_STATE.pipeline = None
    cbi._BOOTSTRAP_STATE.sentinel_manager = None
    cbi._BOOTSTRAP_STATE.active_bootstrap_guard = None
    cbi._BOOTSTRAP_STATE.owner_depths = {}
    cbi._PREPARE_CALL_INVOCATIONS.clear()

    broker = _StubBroker()
    monkeypatch.setattr(cbi, "_bootstrap_dependency_broker", lambda: broker)
    monkeypatch.setattr(cbi, "read_bootstrap_heartbeat", lambda: False)
    monkeypatch.setattr(cbi, "_resolve_bootstrap_wait_timeout", lambda vector_heavy=False: 0.01)

    prepare_calls = {"count": 0}
    start_event = threading.Event()

    def _fake_prepare_impl(**_kwargs: object) -> tuple[Any, Any]:
        prepare_calls["count"] += 1
        start_event.wait(timeout=0.25)
        pipeline = SimpleNamespace(manager=SimpleNamespace())
        return pipeline, (lambda *_: None)

    monkeypatch.setattr(cbi, "_prepare_pipeline_for_bootstrap_impl", _fake_prepare_impl)

    def _install_trigger(name: str) -> types.ModuleType:
        module = _build_stub_module(name)
        exec(
            "import coding_bot_interface as cbi\n\n"
            "def trigger_bootstrap():\n"
            "    return cbi.prepare_pipeline_for_bootstrap(\n"
            "        pipeline_cls=type('Pipeline', (), {}),\n"
            "        context_builder=None,\n"
            "        bot_registry=None,\n"
            "        data_bot=None,\n"
            "        bootstrap_wait_timeout=0.01,\n"
            "    )\n",
            module.__dict__,
        )
        return module

    modules = [
        _install_trigger("orchestrator_loader"),
        _install_trigger("cognition_layer"),
        _install_trigger("prediction_manager_bot"),
    ]
    for module in modules:
        importlib.invalidate_caches()
        sys.modules[module.__name__] = module

    results: list[tuple[Any, Any]] = []

    def _call_trigger(mod: types.ModuleType) -> None:
        results.append(mod.trigger_bootstrap())

    threads = [threading.Thread(target=_call_trigger, args=(module,)) for module in modules]
    for thread in threads:
        thread.start()

    time.sleep(0.05)
    start_event.set()
    for thread in threads:
        thread.join(timeout=1.0)

    assert prepare_calls["count"] == 1, "only one bootstrap attempt should run"
    assert len({id(pipeline) for pipeline, _promote in results}) == 1

    prepare_logs = [record for record in caplog.records if "calling prepare_pipeline_for_bootstrap" in record.getMessage()]
    assert len(prepare_logs) == 1, "initial prepare log should not repeat"

    assert broker.calls, "dependency broker should be exercised"
    reentry_logs = [record for record in caplog.records if "prepare_pipeline.bootstrap.reentry_block" in record.getMessage()]
    assert reentry_logs, "re-entry guard should handle slow bootstrap without spawning new attempts"
