import sys
import types
import importlib
import pytest


class _DummyBuilder:
    def refresh_db_weights(self):
        return None


def test_module_retirement_service_requires_builder(tmp_path):
    from module_retirement_service import ModuleRetirementService

    ModuleRetirementService(tmp_path, context_builder=_DummyBuilder())
    with pytest.raises(ValueError):
        ModuleRetirementService(tmp_path, context_builder=None)  # type: ignore[arg-type]


def test_scalability_pipeline_requires_builder(monkeypatch):
    class DummyDev:
        def __init__(self, *a, **k):
            pass

    class DummyTester:
        pass

    class DummyScaler:
        pass

    class DummyDeployer:
        db = {}

        def deploy(self, *a, **k):
            return 1

        def __init__(self, *a, **k):
            pass

    monkeypatch.setitem(
        sys.modules,
        "vector_service",
        types.SimpleNamespace(ContextBuilder=_DummyBuilder),
    )
    monkeypatch.setitem(
        sys.modules,
        "menace.bot_development_bot",
        types.SimpleNamespace(BotDevelopmentBot=DummyDev, BotSpec=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "menace.bot_testing_bot",
        types.SimpleNamespace(BotTestingBot=DummyTester),
    )
    monkeypatch.setitem(
        sys.modules,
        "menace.scalability_assessment_bot",
        types.SimpleNamespace(ScalabilityAssessmentBot=DummyScaler, TaskInfo=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "menace.deployment_bot",
        types.SimpleNamespace(DeploymentBot=DummyDeployer, DeploymentSpec=object),
    )

    sp = importlib.import_module("menace.scalability_pipeline")
    pipeline_cls = sp.ScalabilityPipeline
    builder = _DummyBuilder()
    pipeline_cls(
        context_builder=builder,
        developer=DummyDev(),
        tester=DummyTester(),
        scaler=DummyScaler(),
        deployer=DummyDeployer(),
    )

    with pytest.raises(ValueError):
        pipeline_cls(
            context_builder=None,  # type: ignore[arg-type]
            developer=DummyDev(),
            tester=DummyTester(),
            scaler=DummyScaler(),
            deployer=DummyDeployer(),
        )
