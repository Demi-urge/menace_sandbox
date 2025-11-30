from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


def _install_stub(monkeypatch, name: str, **attrs: object) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    for attr, value in attrs.items():
        setattr(module, attr, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _stub_orchestrator_dependencies(monkeypatch) -> None:
    base = sys.modules.get("menace_sandbox")
    if base is None:
        base = types.ModuleType("menace_sandbox")
        base.__path__ = [str(Path(__file__).resolve().parent.parent)]
        base.__spec__ = importlib.machinery.ModuleSpec(
            "menace_sandbox", loader=None, is_package=True
        )
        monkeypatch.setitem(sys.modules, "menace_sandbox", base)

    _install_stub(
        monkeypatch,
        "dynamic_path_router",
        resolve_path=lambda path: Path(path),
        get_project_root=lambda: Path("."),
    )
    _install_stub(
        monkeypatch,
        "bootstrap_timeout_policy",
        compute_prepare_pipeline_component_budgets=lambda: {},
        read_bootstrap_heartbeat=lambda: None,
    )
    _install_stub(
        monkeypatch,
        "db_router",
        DBRouter=type("DBRouter", (), {"__init__": lambda self, *a, **k: None}),
        GLOBAL_ROUTER=None,
    )

    _install_stub(monkeypatch, "menace_sandbox.knowledge_graph", KnowledgeGraph=type("KG", (), {}))
    _install_stub(
        monkeypatch,
        "menace_sandbox.advanced_error_management",
        AutomatedRollbackManager=type("ARM", (), {}),
    )
    _install_stub(
        monkeypatch,
        "menace_sandbox.self_coding_engine",
        SelfCodingEngine=type("SCE", (), {}),
    )
    _install_stub(
        monkeypatch,
        "menace_sandbox.rollback_validator",
        RollbackValidator=type("RV", (), {}),
    )
    _install_stub(
        monkeypatch,
        "menace_sandbox.oversight_bots",
        L1OversightBot=type("L1", (), {}),
        L2OversightBot=type("L2", (), {}),
        L3OversightBot=type("L3", (), {}),
        M1OversightBot=type("M1", (), {}),
        M2OversightBot=type("M2", (), {}),
        M3OversightBot=type("M3", (), {}),
        H1OversightBot=type("H1", (), {}),
        H2OversightBot=type("H2", (), {}),
        H3OversightBot=type("H3", (), {}),
    )
    _install_stub(
        monkeypatch,
        "menace_sandbox.model_automation_pipeline",
        ModelAutomationPipeline=type("Pipeline", (), {}),
        AutomationResult=type("AutomationResult", (), {}),
    )

    coding_state = SimpleNamespace()

    def _dependency_broker():
        return SimpleNamespace(resolve=lambda: (None, None), advertise=lambda **_: None)

    def _advertise_placeholder(
        *, dependency_broker=None, pipeline=None, manager=None, owner=True
    ):
        sentinel = manager or SimpleNamespace()
        pipeline_candidate = pipeline or SimpleNamespace(
            manager=sentinel, bootstrap_placeholder=True
        )
        (dependency_broker or _dependency_broker()).advertise(
            pipeline=pipeline_candidate, sentinel=sentinel, owner=owner
        )
        return pipeline_candidate, sentinel

    _install_stub(
        monkeypatch,
        "menace_sandbox.coding_bot_interface",
        _BOOTSTRAP_STATE=coding_state,
        _bootstrap_dependency_broker=_dependency_broker,
        _GLOBAL_BOOTSTRAP_COORDINATOR=SimpleNamespace(peek_active=lambda: None),
        _current_bootstrap_context=lambda: None,
        _peek_owner_promise=lambda *a, **k: None,
        _resolve_bootstrap_wait_timeout=lambda *a, **k: None,
        prepare_pipeline_for_bootstrap=lambda **_k: (SimpleNamespace(manager=None), lambda *_a: None),
        advertise_bootstrap_placeholder=_advertise_placeholder,
        read_bootstrap_heartbeat=lambda *a, **k: None,
    )

    _install_stub(
        monkeypatch,
        "menace_sandbox.discrepancy_detection_bot",
        DiscrepancyDetectionBot=type("DiscrepancyDetectionBot", (), {}),
    )
    _install_stub(
        monkeypatch,
        "menace_sandbox.efficiency_bot",
        EfficiencyBot=type("EfficiencyBot", (), {}),
    )
    _install_stub(
        monkeypatch,
        "menace_sandbox.neuroplasticity",
        Outcome=type("Outcome", (), {}),
        PathwayDB=type("PathwayDB", (), {}),
        PathwayRecord=type("PathwayRecord", (), {}),
    )
    _install_stub(monkeypatch, "menace_sandbox.ad_integration", AdIntegration=type("AdIntegration", (), {}))
    _install_stub(
        monkeypatch,
        "menace_sandbox.watchdog",
        Watchdog=type("Watchdog", (), {"__init__": lambda self, *a, **k: None}),
        ContextBuilder=object,
    )
    _install_stub(
        monkeypatch,
        "menace_sandbox.error_bot",
        ErrorDB=type("ErrorDB", (), {"__init__": lambda self, *a, **k: None}),
    )
    _install_stub(
        monkeypatch,
        "menace_sandbox.resource_allocation_optimizer",
        ROIDB=type("ROIDB", (), {"__init__": lambda self, *a, **k: None}),
    )
    _install_stub(
        monkeypatch,
        "menace_sandbox.data_bot",
        MetricsDB=type("MetricsDB", (), {"__init__": lambda self, *a, **k: None}),
    )
    _install_stub(monkeypatch, "menace_sandbox.trending_scraper", TrendingScraper=type("TrendingScraper", (), {}))
    _install_stub(monkeypatch, "menace_sandbox.self_learning_service", main=lambda *a, **k: None)
    _install_stub(
        monkeypatch,
        "menace_sandbox.strategic_planner",
        StrategicPlanner=type(
            "StrategicPlanner", (), {"__init__": lambda self, *a, **k: None}
        ),
    )
    _install_stub(
        monkeypatch,
        "menace_sandbox.strategy_prediction_bot",
        StrategyPredictionBot=type("StrategyPredictionBot", (), {}),
    )
    _install_stub(monkeypatch, "menace_sandbox.autoscaler", Autoscaler=type("Autoscaler", (), {}))
    _install_stub(monkeypatch, "menace_sandbox.trend_predictor", TrendPredictor=type("TrendPredictor", (), {}))
    _install_stub(monkeypatch, "menace_sandbox.identity_seeder", seed_identity=lambda *a, **k: None)
    _install_stub(monkeypatch, "menace_sandbox.session_vault", SessionVault=type("SessionVault", (), {}))
    _install_stub(
        monkeypatch,
        "menace_sandbox.cognition_layer",
        build_cognitive_context=lambda *a, **k: (None, ""),
        log_feedback=lambda *a, **k: None,
    )


class _Broker:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.calls: list[int] = []
        self.advertised: list[dict[str, object | None]] = []

    def resolve(self):
        self.calls.append(len(self.calls))
        if len(self.calls) > 1:
            return self.pipeline, getattr(self.pipeline, "manager", None)
        return None, None

    def advertise(self, **kwargs):
        self.advertised.append(kwargs)
        return None


class _Dummy:
    def __init__(self, *args, **kwargs):
        pass


def test_orchestrator_reuses_active_bootstrap_pipeline(monkeypatch):
    """An active guard promise should block new orchestrator bootstrap attempts."""

    _stub_orchestrator_dependencies(monkeypatch)
    module = importlib.import_module("menace_sandbox.menace_orchestrator")

    guard_token = object()
    fake_pipeline = SimpleNamespace(manager=mock.Mock(name="manager"))
    broker = _Broker(fake_pipeline)

    context_builder = SimpleNamespace(refresh_db_weights=lambda: None)
    router = SimpleNamespace()

    monkeypatch.setattr(module, "_BOOTSTRAP_STATE", SimpleNamespace(active_bootstrap_guard=guard_token))
    monkeypatch.setattr(module, "_peek_owner_promise", lambda *_a, **_k: object())
    monkeypatch.setattr(module, "_resolve_bootstrap_wait_timeout", lambda *_a, **_k: 0.05)
    monkeypatch.setattr(module, "_bootstrap_dependency_broker", lambda: broker)
    monkeypatch.setattr(module, "_current_bootstrap_context", lambda: None)
    monkeypatch.setattr(module, "read_bootstrap_heartbeat", lambda: None)
    monkeypatch.setattr(
        module,
        "prepare_pipeline_for_bootstrap",
        mock.Mock(side_effect=AssertionError("prepare_pipeline_for_bootstrap should not run")),
    )
    monkeypatch.setattr(module, "compute_prepare_pipeline_component_budgets", lambda: {})

    for attr in (
        "KnowledgeGraph",
        "DiscrepancyDetectionBot",
        "EfficiencyBot",
        "Watchdog",
        "ErrorDB",
        "ROIDB",
        "MetricsDB",
        "StrategicPlanner",
        "StrategyPredictionBot",
        "Autoscaler",
        "TrendPredictor",
    ):
        monkeypatch.setattr(module, attr, _Dummy)

    orchestrator = module.MenaceOrchestrator(
        context_builder=context_builder,
        router=router,
        auto_bootstrap=False,
        ad_client=_Dummy(),
    )

    assert len(broker.calls) >= 1
    assert orchestrator.pipeline is not None
    assert broker.advertised
    module.prepare_pipeline_for_bootstrap.assert_not_called()


def test_orchestrator_waits_on_bootstrap_heartbeat(monkeypatch):
    """A bootstrap heartbeat should trigger wait/backoff before new bootstrap."""

    _stub_orchestrator_dependencies(monkeypatch)
    module = importlib.import_module("menace_sandbox.menace_orchestrator")

    fake_pipeline = SimpleNamespace(manager=mock.Mock(name="manager"))

    class _DelayedBroker(_Broker):
        def resolve(self):
            self.calls.append(len(self.calls))
            if len(self.calls) >= 3:
                return self.pipeline, getattr(self.pipeline, "manager", None)
            return None, None

    broker = _DelayedBroker(fake_pipeline)

    context_builder = SimpleNamespace(refresh_db_weights=lambda: None)
    router = SimpleNamespace()

    monkeypatch.setattr(module, "_BOOTSTRAP_STATE", SimpleNamespace(active_bootstrap_guard=None))
    monkeypatch.setattr(module, "_peek_owner_promise", lambda *_a, **_k: None)
    monkeypatch.setattr(module, "_resolve_bootstrap_wait_timeout", lambda *_a, **_k: 0.2)
    monkeypatch.setattr(module, "_bootstrap_dependency_broker", lambda: broker)
    monkeypatch.setattr(module, "_current_bootstrap_context", lambda: None)
    monkeypatch.setattr(module, "read_bootstrap_heartbeat", lambda *_, **__: {"active": True})
    monkeypatch.setattr(
        module,
        "prepare_pipeline_for_bootstrap",
        mock.Mock(side_effect=AssertionError("prepare_pipeline_for_bootstrap should not run")),
    )
    monkeypatch.setattr(module, "compute_prepare_pipeline_component_budgets", lambda: {})

    for attr in (
        "KnowledgeGraph",
        "DiscrepancyDetectionBot",
        "EfficiencyBot",
        "Watchdog",
        "ErrorDB",
        "ROIDB",
        "MetricsDB",
        "StrategicPlanner",
        "StrategyPredictionBot",
        "Autoscaler",
        "TrendPredictor",
    ):
        monkeypatch.setattr(module, attr, _Dummy)

    orchestrator = module.MenaceOrchestrator(
        context_builder=context_builder,
        router=router,
        auto_bootstrap=False,
        ad_client=_Dummy(),
    )

    assert len(broker.calls) >= 1
    assert orchestrator.pipeline is not None
    assert broker.advertised, "orchestrator should advertise the resolved pipeline early"
    module.prepare_pipeline_for_bootstrap.assert_not_called()


def test_orchestrator_reuses_broker_placeholder_promise(monkeypatch):
    _stub_orchestrator_dependencies(monkeypatch)
    module = importlib.import_module("menace_sandbox.menace_orchestrator")

    placeholder_pipeline = SimpleNamespace(
        manager=mock.Mock(name="manager"), bootstrap_placeholder=True
    )
    broker = _Broker(placeholder_pipeline)
    broker.active_owner = True

    class _Promise:
        def __init__(self, pipeline):
            self.pipeline = pipeline
            self.done = True
            self._event = SimpleNamespace(wait=lambda timeout=None: None)

        def wait(self):
            return self.pipeline, lambda *_a, **_k: None

    promise = _Promise(placeholder_pipeline)

    context_builder = SimpleNamespace(refresh_db_weights=lambda: None)
    router = SimpleNamespace()

    monkeypatch.setattr(module, "_BOOTSTRAP_STATE", SimpleNamespace(active_bootstrap_guard=None))
    monkeypatch.setattr(module, "_bootstrap_dependency_broker", lambda: broker)
    monkeypatch.setattr(module, "_current_bootstrap_context", lambda: None)
    monkeypatch.setattr(module, "read_bootstrap_heartbeat", lambda: None)
    monkeypatch.setattr(module, "_peek_owner_promise", lambda *_a, **_k: None)
    monkeypatch.setattr(module, "_resolve_bootstrap_wait_timeout", lambda *_a, **_k: 0.1)
    monkeypatch.setattr(
        module,
        "_GLOBAL_BOOTSTRAP_COORDINATOR",
        SimpleNamespace(peek_active=lambda: promise),
    )
    monkeypatch.setattr(
        module,
        "prepare_pipeline_for_bootstrap",
        mock.Mock(side_effect=AssertionError("prepare_pipeline_for_bootstrap should not run")),
    )
    monkeypatch.setattr(module, "compute_prepare_pipeline_component_budgets", lambda: {})

    for attr in (
        "KnowledgeGraph",
        "DiscrepancyDetectionBot",
        "EfficiencyBot",
        "Watchdog",
        "ErrorDB",
        "ROIDB",
        "MetricsDB",
        "StrategicPlanner",
        "StrategyPredictionBot",
        "Autoscaler",
        "TrendPredictor",
    ):
        monkeypatch.setattr(module, attr, _Dummy)

    orchestrator = module.MenaceOrchestrator(
        context_builder=context_builder,
        router=router,
        auto_bootstrap=False,
        ad_client=_Dummy(),
    )

    assert orchestrator.pipeline is not None
    module.prepare_pipeline_for_bootstrap.assert_not_called()
    assert any(entry.get("owner") is True for entry in broker.advertised)


def test_orchestrator_advertises_placeholder_before_prepare(monkeypatch):
    _stub_orchestrator_dependencies(monkeypatch)
    placeholders: list[tuple[object, object, bool]] = []

    class _RecordingBroker:
        def __init__(self) -> None:
            self.advertised: list[dict[str, object | None]] = []

        def resolve(self) -> tuple[None, None]:
            return None, None

        def advertise(self, **kwargs: object) -> None:
            self.advertised.append(kwargs)

    broker = _RecordingBroker()

    def _advertise_bootstrap_placeholder(
        *, dependency_broker=None, pipeline=None, manager=None, owner=True
    ):
        sentinel = manager or SimpleNamespace(owner_placeholder=True)
        pipeline_candidate = pipeline or SimpleNamespace(
            manager=sentinel, bootstrap_placeholder=True
        )
        (dependency_broker or broker).advertise(
            pipeline=pipeline_candidate, sentinel=sentinel, owner=owner
        )
        placeholders.append((pipeline_candidate, sentinel, owner))
        return pipeline_candidate, sentinel

    import menace_sandbox.coding_bot_interface as cbi

    monkeypatch.setattr(cbi, "advertise_bootstrap_placeholder", _advertise_bootstrap_placeholder)
    monkeypatch.setattr(cbi, "_bootstrap_dependency_broker", lambda: broker)
    monkeypatch.setattr(cbi, "_GLOBAL_BOOTSTRAP_COORDINATOR", SimpleNamespace(peek_active=lambda: None))
    monkeypatch.setattr(cbi, "_peek_owner_promise", lambda *_a, **_k: None)
    monkeypatch.setattr(cbi, "_current_bootstrap_context", lambda: None)
    monkeypatch.setattr(cbi, "_resolve_bootstrap_wait_timeout", lambda *_a, **_k: None)
    monkeypatch.setattr(cbi, "read_bootstrap_heartbeat", lambda *_a, **_k: None)

    prepare_calls: dict[str, object] = {}

    def _prepare_pipeline_for_bootstrap(**_kwargs: object):
        prepare_calls["pre_advertisements"] = list(broker.advertised)
        assert broker.advertised, "placeholder should be advertised before prepare"
        assert broker.advertised[-1].get("owner") is True
        pipeline = SimpleNamespace(manager=None)

        def _promote(manager: object) -> None:
            prepare_calls["promoted"] = manager

        return pipeline, _promote

    monkeypatch.setattr(cbi, "prepare_pipeline_for_bootstrap", _prepare_pipeline_for_bootstrap)

    module = importlib.reload(importlib.import_module("menace_sandbox.menace_orchestrator"))

    context_builder = SimpleNamespace(refresh_db_weights=lambda: None)
    orchestrator = module.MenaceOrchestrator(
        context_builder=context_builder,
        auto_bootstrap=False,
        ad_client=SimpleNamespace(),
    )

    assert placeholders, "placeholder advertisement should run"
    if "pre_advertisements" in prepare_calls:
        assert prepare_calls["pre_advertisements"]
        assert any(entry.get("owner") for entry in prepare_calls["pre_advertisements"])
    else:
        assert broker.advertised
        assert any(entry.get("owner") for entry in broker.advertised)

    manager = object()
    orchestrator.promote_pipeline_manager(manager)
    assert any(entry.get("sentinel") is manager for entry in broker.advertised)
