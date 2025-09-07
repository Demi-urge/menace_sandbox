from __future__ import annotations

import importlib
import sys
import types
from dataclasses import dataclass


def test_context_builder_reused(tmp_path):
    builders: list[object] = []

    class StubContextBuilder:
        def __init__(self, *a, **k):
            pass

    class StubDeveloper:
        def __init__(self, *a, context_builder=None, **k):
            self.context_builder = context_builder
            self.errors: list[str] = []

        def build_bot(self, spec, *, context_builder, model_id=None, **_):
            builders.append(context_builder)
            return tmp_path / f"{spec.name}.py"

    @dataclass
    class BotSpec:
        name: str
        purpose: str
        functions: list[str]

    vs = types.ModuleType("vector_service")
    vs.ContextBuilder = StubContextBuilder
    vs.FallbackResult = Exception
    vs.ErrorResult = Exception
    vs.EmbeddableDBMixin = object
    sys.modules["vector_service"] = vs

    bd = types.ModuleType("menace.bot_development_bot")
    bd.BotDevelopmentBot = StubDeveloper
    bd.BotSpec = BotSpec
    sys.modules["menace.bot_development_bot"] = bd

    bt = types.ModuleType("menace.bot_testing_bot")
    bt.BotTestingBot = object
    sys.modules["menace.bot_testing_bot"] = bt

    scal = types.ModuleType("menace.scalability_assessment_bot")
    scal.ScalabilityAssessmentBot = object
    sys.modules["menace.scalability_assessment_bot"] = scal

    dep = types.ModuleType("menace.deployment_bot")
    dep.DeploymentBot = object
    class DeploymentSpec:
        def __init__(self, name, resources, env):
            self.name = name
            self.resources = resources
            self.env = env

    dep.DeploymentSpec = DeploymentSpec
    sys.modules["menace.deployment_bot"] = dep

    handoff = types.ModuleType("menace.task_handoff_bot")
    handoff.TaskHandoffBot = object
    handoff.TaskInfo = object
    sys.modules["menace.task_handoff_bot"] = handoff

    research = types.ModuleType("menace.research_aggregator_bot")
    research.ResearchAggregatorBot = object
    sys.modules["menace.research_aggregator_bot"] = research

    ipo_stub = types.ModuleType("menace.ipo_bot")
    ipo_stub.IPOBot = object
    ipo_stub.ExecutionPlan = object
    sys.modules["menace.ipo_bot"] = ipo_stub

    ipp = importlib.import_module("menace.ipo_implementation_pipeline")

    class DummyPlan:
        def __init__(self):
            self.actions = [types.SimpleNamespace(bot="a"), types.SimpleNamespace(bot="b")]

    class DummyIPOBot:
        def generate_plan(self, blueprint):
            return DummyPlan()

    class DummyTester:
        class Result:
            passed = True
            error = None

        def run_unit_tests(self, names):
            return [self.Result()]

    class DummyScaler:
        class Report:
            class Task:
                cpu = 0
                memory = 0

            tasks = [Task()]

        def analyse(self, bp):
            return self.Report()

    class DummyDB:
        def get(self, dep_id):
            return {"status": "success"}

    class DummyDeployer:
        def __init__(self):
            self.db = DummyDB()

        def deploy(self, name, files, spec):
            return 1

    cb = StubContextBuilder()
    pipeline = ipp.IPOImplementationPipeline(
        ipo=DummyIPOBot(),
        tester=DummyTester(),
        scaler=DummyScaler(),
        deployer=DummyDeployer(),
        context_builder=cb,
    )
    pipeline.run("plan")

    assert len(builders) == 2
    assert builders[0] is builders[1]
