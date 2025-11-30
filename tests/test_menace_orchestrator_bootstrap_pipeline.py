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

    _install_stub(
        monkeypatch,
        "menace_sandbox.coding_bot_interface",
        _BOOTSTRAP_STATE=coding_state,
        _bootstrap_dependency_broker=_dependency_broker,
        _current_bootstrap_context=lambda: None,
        _peek_owner_promise=lambda *a, **k: None,
        _resolve_bootstrap_wait_timeout=lambda *a, **k: None,
        prepare_pipeline_for_bootstrap=lambda **_k: (SimpleNamespace(manager=None), lambda *_a: None),
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
    _install_stub(monkeypatch, "menace_sandbox.watchdog", Watchdog=type("Watchdog", (), {}), ContextBuilder=object)
    _install_stub(monkeypatch, "menace_sandbox.error_bot", ErrorDB=type("ErrorDB", (), {}))
    _install_stub(monkeypatch, "menace_sandbox.resource_allocation_optimizer", ROIDB=type("ROIDB", (), {}))
    _install_stub(monkeypatch, "menace_sandbox.data_bot", MetricsDB=type("MetricsDB", (), {}))
    _install_stub(monkeypatch, "menace_sandbox.trending_scraper", TrendingScraper=type("TrendingScraper", (), {}))
    _install_stub(monkeypatch, "menace_sandbox.self_learning_service", main=lambda *a, **k: None)
    _install_stub(
        monkeypatch,
        "menace_sandbox.strategic_planner",
        StrategicPlanner=type("StrategicPlanner", (), {}),
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

    assert len(broker.calls) >= 2
    assert orchestrator.pipeline is fake_pipeline
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

    assert len(broker.calls) >= 3
    assert orchestrator.pipeline is fake_pipeline
    assert broker.advertised, "orchestrator should advertise the resolved pipeline early"
    module.prepare_pipeline_for_bootstrap.assert_not_called()
