import types

import pytest

import tests.test_self_improvement_engine as sie_tests
import menace.self_improvement.engine as sie


def test_engine_bootstrap_pipeline_promoter(monkeypatch, caplog):
    builder = sie_tests.DummyBuilder()
    pipeline = types.SimpleNamespace(manager="sentinel", context_builder=builder)
    promote_calls: list[object] = []

    class _DummyMetricsDB:
        def __init__(self, *_, **__):
            pass

    class _DummyErrorDB:
        def __init__(self, *_, **__):
            pass

    class _StopInit(Exception):
        pass

    def _halting_error_bot(*_a, **_k):
        raise _StopInit()

    class _DummyAggregator:
        def __init__(self, *_, **kwargs):
            self.context_builder = kwargs.get("context_builder")

    class _DummySandboxSettings:
        def __init__(self, *_, **__):
            self.baseline_window = 5
            self.energy_deviation = 1.0
            self.gpt_memory_db = "memory.db"
            self.synergy_learner = "default"
            self.roi = types.SimpleNamespace(stagnation_cycles=3)

    class _DummyAdaptiveROIPredictor:
        def __init__(self, *_, **__):
            pass

    class _DummyCompositeWorkflowScorer:
        def __init__(self, *_, **__):
            pass

    class _DummySynergyLearner:
        def __init__(self, *_, **__):
            self.weights = {
                "roi": 1.0,
                "efficiency": 1.0,
                "resilience": 1.0,
                "antifragility": 1.0,
                "reliability": 1.0,
                "maintainability": 1.0,
                "throughput": 1.0,
            }

    class _DummySettings:
        synergy_weight_file = "weights.json"
        synergy_weight_checkpoint = "weights.json"
        adaptive_roi_prioritization = False
        adaptive_roi_train_interval = 3600
        borderline_confidence_threshold = 0.5
        synergy_weight_roi = 1.0
        synergy_weight_efficiency = 1.0
        synergy_weight_resilience = 1.0
        synergy_weight_antifragility = 1.0
        synergy_weight_reliability = 1.0
        synergy_weight_maintainability = 1.0
        synergy_weight_throughput = 1.0

        def __getattr__(self, name):
            if name.endswith("_file") or name.endswith("_path"):
                return "stub"
            if name.endswith("_mode"):
                return ""
            return 1.0

    monkeypatch.setattr(sie, "ResearchAggregatorBot", _DummyAggregator)
    monkeypatch.setattr(sie, "DiagnosticManager", _DummyAggregator)
    monkeypatch.setattr(sie, "SandboxSettings", _DummySandboxSettings)
    monkeypatch.setattr(sie, "AdaptiveROIPredictor", _DummyAdaptiveROIPredictor)
    monkeypatch.setattr(sie, "CompositeWorkflowScorer", _DummyCompositeWorkflowScorer)
    monkeypatch.setattr(sie, "SynergyWeightLearner", _DummySynergyLearner)
    monkeypatch.setattr(sie, "settings", _DummySettings())
    monkeypatch.setattr(sie, "MetricsDB", _DummyMetricsDB)
    monkeypatch.setattr(sie, "ErrorDB", _DummyErrorDB)
    monkeypatch.setattr(sie, "ErrorBot", _halting_error_bot)

    def _fake_prepare(**kwargs):
        def _promote(manager):
            promote_calls.append(manager)
            pipeline.manager = manager

        return pipeline, _promote

    monkeypatch.setattr(sie, "prepare_pipeline_for_bootstrap", _fake_prepare, raising=False)
    engine = sie.SelfImprovementEngine.__new__(sie.SelfImprovementEngine)
    with pytest.raises(_StopInit):
        sie.SelfImprovementEngine.__init__(
            engine,
            context_builder=builder,
            synergy_learner_cls=_DummySynergyLearner,
        )
    assert engine.pipeline is pipeline
    assert callable(engine.pipeline_promoter)
    engine.promote_pipeline_manager("manager")
    assert pipeline.manager == "manager"
    assert promote_calls == ["manager"]
    assert engine.pipeline_promoter is None
    assert "re-entrant initialisation depth" not in caplog.text
