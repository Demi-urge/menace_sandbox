from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
import types

import pytest


def _install_module_stub(monkeypatch, name: str, **attrs):
    module = types.ModuleType(name)
    for attr, value in attrs.items():
        setattr(module, attr, value)
    for prefix in ("", "menace.", "menace_sandbox."):
        qualified = f"{prefix}{name}" if prefix else name
        monkeypatch.setitem(sys.modules, qualified, module)
    return module


@pytest.fixture(autouse=True)
def _stub_code_database(monkeypatch):
    _install_module_stub(
        monkeypatch,
        "code_database",
        CodeDB=type("CodeDB", (), {}),
        PatchHistoryDB=type("PatchHistoryDB", (), {}),
        PatchRecord=type("PatchRecord", (), {}),
    )
    yield


def _install_prepare_stub(monkeypatch, module):
    calls: list[dict[str, object]] = []

    def _prepare(**kwargs):
        sentinel = SimpleNamespace(name="sentinel")
        helpers = [SimpleNamespace(manager=sentinel), SimpleNamespace(manager=sentinel, initial_manager=sentinel)]
        pipeline = SimpleNamespace(manager=sentinel, helpers=helpers, context_builder=kwargs.get("context_builder"))

        def _promote(manager, *, extra_sentinels=None):
            pipeline.manager = manager
            for helper in pipeline.helpers:
                helper.manager = manager
                if hasattr(helper, "initial_manager"):
                    helper.initial_manager = manager

        calls.append({"kwargs": kwargs, "pipeline": pipeline, "promoter": _promote})
        return pipeline, _promote

    monkeypatch.setattr(module, "prepare_pipeline_for_bootstrap", _prepare)
    return calls


def _patch_menace_orchestrator_dependencies(monkeypatch, module):
    simple_bot = lambda *a, **k: SimpleNamespace()
    for name in [
        "L1OversightBot",
        "L2OversightBot",
        "L3OversightBot",
        "M1OversightBot",
        "M2OversightBot",
        "M3OversightBot",
        "H1OversightBot",
        "H2OversightBot",
        "H3OversightBot",
    ]:
        monkeypatch.setattr(module, name, simple_bot)

    class _StubKG:
        def root_causes(self, name):
            return []

    monkeypatch.setattr(module, "KnowledgeGraph", lambda: _StubKG())
    monkeypatch.setattr(module, "AutomatedRollbackManager", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(module, "SelfCodingEngine", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(module, "RollbackValidator", lambda *a, **k: SimpleNamespace())

    class _StubWatchdog:
        def __init__(self, *_, **__):
            self.healer = SimpleNamespace(heal=lambda *a, **k: None)

        def record_heartbeat(self, *_, **__):
            return None

        def schedule(self, *_, **__):
            return None

    monkeypatch.setattr(module, "Watchdog", _StubWatchdog)
    monkeypatch.setattr(module, "ErrorDB", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(module, "ROIDB", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(module, "MetricsDB", lambda *a, **k: SimpleNamespace())

    class _StubPlanner:
        def __init__(self, *_, **__):
            self.autoscaler = SimpleNamespace(scale=lambda *a, **k: None)

        def plan_cycle(self):
            return ""

    monkeypatch.setattr(module, "StrategicPlanner", _StubPlanner)
    monkeypatch.setattr(module, "StrategyPredictionBot", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(module, "Autoscaler", lambda *a, **k: SimpleNamespace(scale=lambda *a, **k: None))
    monkeypatch.setattr(module, "TrendPredictor", lambda *a, **k: SimpleNamespace())

    class _StubDiscrepancy:
        severity_threshold = 1.0

        def scan(self):
            return []

    monkeypatch.setattr(module, "DiscrepancyDetectionBot", lambda *a, **k: _StubDiscrepancy())
    monkeypatch.setattr(module, "EfficiencyBot", lambda *a, **k: SimpleNamespace(assess_efficiency=lambda: {}))

    class _StubAd:
        async def process_sales_async(self):
            return None

    monkeypatch.setattr(module, "AdIntegration", lambda *a, **k: _StubAd())

    class _StubRouter:
        def __init__(self, *_, **__):
            self.conn = None

    monkeypatch.setattr(module, "DBRouter", _StubRouter)
    monkeypatch.setattr(module, "resolve_path", lambda path: Path(str(path)))


def _patch_experiment_manager_dependencies(monkeypatch, module):
    monkeypatch.setattr(module, "ExperimentHistoryDB", lambda *a, **k: SimpleNamespace(add=lambda *b, **c: None, variant_values=lambda *_: []))
    monkeypatch.setattr(module, "MutationLineage", lambda *a, **k: SimpleNamespace(history_db=None))


def _preinstall_self_improvement_modules(monkeypatch):
    _install_module_stub(
        monkeypatch,
        "error_bot",
        ErrorBot=lambda *a, **k: SimpleNamespace(patch_db=None),
        ErrorDB=lambda *a, **k: SimpleNamespace(),
    )
    _install_module_stub(
        monkeypatch,
        "composite_workflow_scorer",
        CompositeWorkflowScorer=lambda *a, **k: SimpleNamespace(),
    )


def _patch_self_improvement_dependencies(monkeypatch, module, tmp_path):
    def _safe_set(name: str, value):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, value)

    _safe_set("ResearchAggregatorBot", lambda *a, **k: SimpleNamespace(context_builder=k.get("context_builder")))
    _safe_set("InfoDB", lambda *a, **k: SimpleNamespace())
    _safe_set("DiagnosticManager", lambda *a, **k: SimpleNamespace())
    _safe_set("ErrorBot", lambda *a, **k: SimpleNamespace(patch_db=None))
    _safe_set("ErrorDB", lambda *a, **k: SimpleNamespace())
    _safe_set("MetricsDB", lambda *a, **k: SimpleNamespace())
    _safe_set("LearningEngine", lambda *a, **k: SimpleNamespace(train=lambda: None))
    _safe_set("CapitalManagementBot", lambda *a, **k: SimpleNamespace())
    _safe_set("UnifiedEventBus", lambda *a, **k: SimpleNamespace())
    _safe_set("EvolutionHistoryDB", lambda *a, **k: SimpleNamespace())
    _safe_set("ActionPlanner", lambda *a, **k: SimpleNamespace())
    _safe_set("CompositeWorkflowScorer", lambda *a, **k: SimpleNamespace())
    _safe_set("WorkflowEvolutionManager", lambda *a, **k: SimpleNamespace())
    _safe_set("integrate_orphans", lambda *a, **k: None)
    _safe_set("post_round_orphan_scan", lambda *a, **k: None)
    _safe_set("generate_patch", lambda *a, **k: None)
    _safe_set("BorderlineBucket", lambda *a, **k: SimpleNamespace())
    _safe_set("HumanAlignmentFlagger", lambda *a, **k: SimpleNamespace())
    _safe_set("PromptStrategyManager", lambda *a, **k: SimpleNamespace())
    _safe_set("StrategyAnalytics", lambda *a, **k: SimpleNamespace())
    _safe_set("SnapshotTracker", lambda *a, **k: SimpleNamespace())
    _safe_set("RelevancyMetricsDB", lambda *a, **k: SimpleNamespace())
    map_path = tmp_path / "module_map.json"
    map_path.write_text("{}")
    if hasattr(module, "ModuleIndexDB"):
        monkeypatch.setattr(module, "ModuleIndexDB", lambda *a, **k: SimpleNamespace(path=map_path))
    _safe_set("ROIResultsDB", lambda *a, **k: SimpleNamespace())
    _safe_set("ConfigurableSelfImprovementPolicy", lambda *a, **k: SimpleNamespace(load=lambda *_: None, path=None))
    _safe_set("ROITracker", None)
    _safe_set("AdaptiveROIPredictor", lambda *a, **k: SimpleNamespace())
    _safe_set("ForesightTracker", lambda *a, **k: SimpleNamespace())
    _safe_set("TruthAdapter", lambda *a, **k: SimpleNamespace())
    _safe_set("GPTMemoryInterface", object)
    _safe_set("GPTKnowledgeService", object)
    _safe_set("RelevancyRadar", lambda *a, **k: SimpleNamespace())
    _safe_set("IntentClusterer", lambda *a, **k: SimpleNamespace())
    _safe_set("UniversalRetriever", lambda *a, **k: SimpleNamespace())
    _safe_set(
        "SynergyWeightLearner",
        lambda *a, **k: SimpleNamespace(
            weights={
                "roi": 1.0,
                "efficiency": 1.0,
                "resilience": 1.0,
                "antifragility": 1.0,
                "reliability": 1.0,
                "maintainability": 1.0,
                "throughput": 1.0,
            }
        ),
    )
    _safe_set("ErrorClusterPredictor", lambda *a, **k: SimpleNamespace())
    _safe_set("_data_dir", lambda: tmp_path)
    _safe_set("_repo_path", lambda: tmp_path)
    weights_path = tmp_path / "synergy_weights.json"
    weights_path.write_text("{}")
    _safe_set(
        "SandboxSettings",
        lambda: SimpleNamespace(
            energy_deviation=1.0,
            baseline_window=5,
            roi=SimpleNamespace(
                momentum_window=5,
                momentum_dev_multiplier=1.0,
                baseline_window=5,
                stagnation_cycles=3,
                roi_stagnation_dev_multiplier=1.0,
            ),
            delta_score_deviation=0.0,
            mae_deviation=1.0,
            acc_deviation=1.0,
            roi_deviation=1.0,
            entropy_deviation=1.0,
            pass_rate_deviation=1.0,
            sandbox_auto_map=False,
            sandbox_autodiscover_modules=False,
            synergy_weight_roi=1.0,
            synergy_weight_efficiency=1.0,
            synergy_weight_resilience=1.0,
            synergy_weight_antifragility=1.0,
            synergy_weight_reliability=1.0,
            synergy_weight_maintainability=1.0,
            synergy_weight_throughput=1.0,
            synergy_weight_file=str(weights_path),
            gpt_memory_db=str(tmp_path / "gpt_memory.db"),
            patch_score_backend_url=None,
            auto_train_synergy=False,
        ),
    )
    _safe_set(
        "settings",
        SimpleNamespace(
            roi_weight=1.0,
            momentum_weight=1.0,
            pass_rate_weight=1.0,
            entropy_weight_scale=0.0,
            momentum_weight_scale=0.0,
            borderline_confidence_threshold=0.0,
            adaptive_roi_prioritization=False,
            synergy_weight_roi=1.0,
            synergy_weight_efficiency=1.0,
            synergy_weight_resilience=1.0,
            synergy_weight_antifragility=1.0,
            synergy_weight_reliability=1.0,
            synergy_weight_maintainability=1.0,
            synergy_weight_throughput=1.0,
            delta_score_dev_multiplier=0.0,
            entropy_ceiling_threshold=0.0,
            entropy_ceiling_consecutive=1,
            sandbox_data_dir=str(tmp_path),
            module_synergy_graph_path=str(tmp_path / "module_graph.json"),
            alignment_failure_threshold=1.0,
            alignment_warning_threshold=1.0,
            alignment_flags_path=str(tmp_path / "alignment_flags.json"),
            recursive_isolated=False,
            recursive_orphan_scan=False,
            momentum_stagnation_dev_multiplier=1.0,
            roi_ema_alpha=0.5,
            synergy_weights_lr=0.1,
            synergy_weight_file=str(weights_path),
            test_redundant_modules=False,
        ),
    )


def test_menace_orchestrator_promotes_bootstrap_pipeline(monkeypatch, caplog):
    oversight_stub = types.ModuleType("menace.oversight_bots")
    for name in [
        "L1OversightBot",
        "L2OversightBot",
        "L3OversightBot",
        "M1OversightBot",
        "M2OversightBot",
        "M3OversightBot",
        "H1OversightBot",
        "H2OversightBot",
        "H3OversightBot",
    ]:
        setattr(oversight_stub, name, lambda *a, **k: SimpleNamespace())
    monkeypatch.setitem(sys.modules, "menace.oversight_bots", oversight_stub)
    monkeypatch.setitem(sys.modules, "menace_sandbox.oversight_bots", oversight_stub)
    capital_stub = types.ModuleType("menace.capital_management_bot")
    capital_stub.CapitalManagementBot = lambda *a, **k: SimpleNamespace()
    monkeypatch.setitem(sys.modules, "menace.capital_management_bot", capital_stub)
    monkeypatch.setitem(sys.modules, "menace_sandbox.capital_management_bot", capital_stub)
    monkeypatch.setitem(sys.modules, "capital_management_bot", capital_stub)
    import menace.menace_orchestrator as mo

    _patch_menace_orchestrator_dependencies(monkeypatch, mo)
    prepare_calls = _install_prepare_stub(monkeypatch, mo)
    builder = SimpleNamespace(refresh_db_weights=lambda: None)
    caplog.set_level(logging.WARNING)
    orchestrator = mo.MenaceOrchestrator(context_builder=builder)
    assert prepare_calls
    pipeline = prepare_calls[-1]["pipeline"]
    manager = SimpleNamespace(name="manager")
    orchestrator.promote_pipeline_manager(manager)
    assert pipeline.manager is manager
    for helper in pipeline.helpers:
        assert helper.manager is manager
        if hasattr(helper, "initial_manager"):
            assert helper.initial_manager is manager
    assert "re-entrant initialisation depth" not in caplog.text


def test_experiment_manager_promotes_bootstrap_pipeline(monkeypatch):
    lineage_stub = types.ModuleType("menace.mutation_lineage")
    lineage_stub.MutationLineage = lambda *a, **k: SimpleNamespace(history_db=None)
    monkeypatch.setitem(sys.modules, "menace.mutation_lineage", lineage_stub)
    monkeypatch.setitem(sys.modules, "menace_sandbox.mutation_lineage", lineage_stub)
    monkeypatch.setitem(sys.modules, "mutation_lineage", lineage_stub)
    import menace.experiment_manager as em

    _patch_experiment_manager_dependencies(monkeypatch, em)
    prepare_calls = _install_prepare_stub(monkeypatch, em)
    data_bot = SimpleNamespace(db=SimpleNamespace(fetch=lambda limit=1: SimpleNamespace(empty=True)))
    capital_bot = SimpleNamespace()
    builder = SimpleNamespace(refresh_db_weights=lambda: None, build_context=lambda *_: None)
    manager = em.ExperimentManager(data_bot, capital_bot, context_builder=builder)
    assert prepare_calls
    pipeline = prepare_calls[-1]["pipeline"]
    real_manager = SimpleNamespace()
    manager.promote_pipeline_manager(real_manager)
    assert pipeline.manager is real_manager
    for helper in pipeline.helpers:
        assert helper.manager is real_manager
        if hasattr(helper, "initial_manager"):
            assert helper.initial_manager is real_manager


def test_self_improvement_engine_promotes_bootstrap_pipeline(monkeypatch, tmp_path, caplog):
    knowledge_stub = SimpleNamespace(
        LocalKnowledgeModule=lambda *a, **k: SimpleNamespace(knowledge=SimpleNamespace()),
        init_local_knowledge=lambda *a, **k: SimpleNamespace(memory=SimpleNamespace()),
    )
    monkeypatch.setitem(
        sys.modules,
        "menace_sandbox.local_knowledge_module",
        knowledge_stub,
    )
    _preinstall_self_improvement_modules(monkeypatch)
    sie = importlib.reload(importlib.import_module("menace.self_improvement.engine"))
    _patch_self_improvement_dependencies(monkeypatch, sie, tmp_path)
    prepare_calls = _install_prepare_stub(monkeypatch, sie)
    builder = SimpleNamespace(refresh_db_weights=lambda: None)
    caplog.set_level(logging.WARNING)
    engine = sie.SelfImprovementEngine(context_builder=builder)
    assert prepare_calls
    pipeline = prepare_calls[-1]["pipeline"]
    real_manager = SimpleNamespace()
    engine.promote_pipeline_manager(real_manager)
    assert pipeline.manager is real_manager
    for helper in pipeline.helpers:
        assert helper.manager is real_manager
        if hasattr(helper, "initial_manager"):
            assert helper.initial_manager is real_manager
    assert "re-entrant initialisation depth" not in caplog.text
