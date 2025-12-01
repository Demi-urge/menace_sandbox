import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

import coding_bot_interface as cbi
from tests.integration.test_bootstrap_import_reuse import (
    _install_cognition_stubs,
    _install_orchestrator_stubs,
    _install_research_stubs,
)

if "menace_sandbox" not in sys.modules:
    pkg_stub = types.ModuleType("menace_sandbox")
    pkg_stub.__path__ = [str(Path(__file__).resolve().parents[2])]
    sys.modules["menace_sandbox"] = pkg_stub


class SpyBroker:
    def __init__(self) -> None:
        self.pipeline = SimpleNamespace(bootstrap_placeholder=True)
        self.manager = SimpleNamespace(bootstrap_placeholder=True)
        self.active_owner = True
        self.advertise_calls: list[tuple[object, object, bool]] = []
        self.resolve_calls = 0

    def advertise(
        self,
        *,
        pipeline: object | None = None,
        sentinel: object | None = None,
        owner: bool = True,
    ) -> tuple[object, object]:
        if pipeline is not None:
            self.pipeline = pipeline
        if sentinel is not None:
            self.manager = sentinel
        self.active_owner = bool(owner)
        self.advertise_calls.append((self.pipeline, self.manager, owner))
        return self.pipeline, self.manager

    def resolve(self) -> tuple[object, object]:
        self.resolve_calls += 1
        return self.pipeline, self.manager


@pytest.fixture
def cascade_bootstrap(monkeypatch: pytest.MonkeyPatch) -> tuple[SpyBroker, list[dict[str, object]]]:
    _install_research_stubs()
    _install_cognition_stubs()
    _install_orchestrator_stubs()

    broker = SpyBroker()

    def _advertise_bootstrap_placeholder(
        *, dependency_broker: SpyBroker | None = None, pipeline=None, manager=None, owner: bool = True
    ) -> tuple[object, object]:
        active_broker = dependency_broker or broker
        return active_broker.advertise(pipeline=pipeline, sentinel=manager, owner=owner)

    def _resolve_bootstrap_placeholders(**_: object) -> tuple[object, object, SpyBroker]:
        pipeline, manager = broker.resolve()
        return pipeline, manager, broker

    readiness_stub = types.SimpleNamespace(
        await_ready=lambda timeout=None: None,  # noqa: ARG005 - signature mirror
        describe=lambda: "ready",
    )

    monkeypatch.setattr(cbi, "_bootstrap_dependency_broker", lambda: broker)
    monkeypatch.setattr(cbi, "advertise_bootstrap_placeholder", _advertise_bootstrap_placeholder)
    monkeypatch.setattr(cbi, "get_active_bootstrap_pipeline", broker.resolve)

    monkeypatch.setitem(
        sys.modules,
        "bootstrap_readiness",
        types.SimpleNamespace(readiness_signal=lambda: readiness_stub),
    )
    monkeypatch.setitem(
        sys.modules,
        "bootstrap_gate",
        types.SimpleNamespace(resolve_bootstrap_placeholders=_resolve_bootstrap_placeholders),
    )

    # Seed a placeholder so the broker already has an owner when modules import.
    broker.advertise(owner=True)

    prepare_calls: list[dict[str, object]] = []

    def _prepare_pipeline_for_bootstrap(**kwargs: object):
        prepare_calls.append(dict(kwargs))
        cbi.logger.info("calling prepare_pipeline_for_bootstrap")
        return broker.pipeline, (lambda *_a, **_k: None)

    monkeypatch.setattr(cbi, "prepare_pipeline_for_bootstrap", _prepare_pipeline_for_bootstrap)
    monkeypatch.setitem(sys.modules, "menace_sandbox.coding_bot_interface", cbi)

    return broker, prepare_calls


@pytest.mark.integration
@pytest.mark.parametrize(
    "import_order",
    [
        (
            "menace_sandbox.research_aggregator_bot",
            "cognition_layer",
            "menace_sandbox.prediction_manager_bot",
            "menace_sandbox.menace_orchestrator",
        ),
        (
            "menace_sandbox.menace_orchestrator",
            "menace_sandbox.prediction_manager_bot",
            "cognition_layer",
            "menace_sandbox.research_aggregator_bot",
        ),
    ],
)
@pytest.mark.usefixtures("_reset_bootstrap_state")
def test_cascade_reuse_single_prepare(import_order, cascade_bootstrap, caplog):
    broker, prepare_calls = cascade_bootstrap

    caplog.set_level("INFO", logger=cbi.logger.name)

    for module_name in import_order:
        sys.modules.pop(module_name, None)
        module = importlib.import_module(module_name)
        if module_name == "menace_sandbox.menace_orchestrator":
            module.MenaceOrchestrator(context_builder=SimpleNamespace())

    prepare_logs = [
        record for record in caplog.records if "calling prepare_pipeline_for_bootstrap" in record.getMessage()
    ]

    assert len(prepare_calls) == 1, "only the first bootstrap caller should prepare the pipeline"
    assert len(prepare_logs) == 1, "prepare call log should be emitted exactly once"

    assert broker.active_owner, "broker owner should remain active for reuse"
    assert all(call[0] is broker.pipeline for call in broker.advertise_calls)
    assert all(call[1] is broker.manager for call in broker.advertise_calls)
    assert broker.resolve_calls >= len(import_order)

    distinct_pipelines = {id(call[0]) for call in broker.advertise_calls}
    assert len(distinct_pipelines) == 1, "modules must not bypass broker with bespoke pipelines"
