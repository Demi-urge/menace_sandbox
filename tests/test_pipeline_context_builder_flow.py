import contextlib
import importlib
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path


def test_builder_flows_through_pipeline(tmp_path, monkeypatch):
    calls: list[str] = []

    class DummyBuilder:
        def build(self, query: str) -> str:
            calls.append(query)
            return ""

    vec_stub = types.ModuleType("vector_service")
    vec_stub.ContextBuilder = object
    vec_stub.FallbackResult = Exception
    vec_stub.ErrorResult = Exception
    vec_stub.EmbeddableDBMixin = object
    sys.modules["vector_service"] = vec_stub

    stub_dr = types.ModuleType("db_router")
    stub_dr.DBRouter = object
    stub_dr.GLOBAL_ROUTER = None
    stub_dr.init_db_router = lambda *a, **k: None
    stub_dr.LOCAL_TABLES = set()
    stub_dr.SHARED_TABLES = set()
    stub_dr.queue_insert = lambda *a, **k: None
    sys.modules["db_router"] = stub_dr
    sys.modules["menace.db_router"] = stub_dr

    thb_mod = types.ModuleType("menace.task_handoff_bot")

    @dataclass
    class TaskInfo:
        name: str
        dependencies: list
        resources: dict
        schedule: str
        code: str
        metadata: dict

    @dataclass
    class TaskPackage:
        tasks: list[TaskInfo]

    class StubHandoffBot:
        def compile(self, tasks):
            return TaskPackage(list(tasks))

        def store_plan(self, tasks):
            pass

        def send_package(self, package):
            pass

    thb_mod.TaskInfo = TaskInfo
    thb_mod.TaskPackage = TaskPackage
    thb_mod.TaskHandoffBot = StubHandoffBot
    sys.modules["menace.task_handoff_bot"] = thb_mod

    iob_mod = types.ModuleType("menace.implementation_optimiser_bot")

    class StubOptimiser:
        def __init__(self, *, context_builder=None):
            self.context_builder = context_builder

        def fill_missing(self, pkg):
            return pkg

        def process(self, pkg):
            pass

    iob_mod.ImplementationOptimiserBot = StubOptimiser
    sys.modules["menace.implementation_optimiser_bot"] = iob_mod

    bdb_mod = types.ModuleType("menace.bot_development_bot")

    class StubDeveloper:
        def __init__(self, *, context_builder):
            self.context_builder = context_builder

        def parse_plan(self, plan_json):
            data = json.loads(plan_json)
            return [
                types.SimpleNamespace(
                    name=d["name"],
                    purpose=d.get("purpose", "demo"),
                    functions=d.get("functions", ["run"]),
                )
                for d in data
            ]

        def _build_prompt(self, spec):
            self.context_builder.build(spec.name)
            return ""

        def build_from_plan(self, plan_json, model_id=None):
            specs = self.parse_plan(plan_json)
            paths = []
            for spec in specs:
                repo_dir = tmp_path / spec.name
                repo_dir.mkdir()
                file_path = repo_dir / f"{spec.name}.py"
                file_path.write_text("")
                self._build_prompt(spec)
                paths.append(file_path)
            return paths

    bdb_mod.BotDevelopmentBot = StubDeveloper
    bdb_mod.BotSpec = types.SimpleNamespace
    sys.modules["menace.bot_development_bot"] = bdb_mod

    research_mod = types.ModuleType("menace.research_aggregator_bot")

    class StubResearcher:
        def __init__(self, requirements=None, *, context_builder=None):
            self.context_builder = context_builder

        def process(self, text):
            pass

    research_mod.ResearchAggregatorBot = StubResearcher
    sys.modules["menace.research_aggregator_bot"] = research_mod

    ipo_mod = types.ModuleType("menace.ipo_bot")

    class StubIPO:
        def __init__(self, *, context_builder=None):
            self.context_builder = context_builder

        def generate_plan(self, blueprint):
            return types.SimpleNamespace(actions=[], graph=types.SimpleNamespace())

    ipo_mod.IPOBot = StubIPO
    sys.modules["menace.ipo_bot"] = ipo_mod

    models_repo = types.ModuleType("menace.models_repo")
    models_repo.clone_to_new_repo = lambda mid: Path(".")
    models_repo.model_build_lock = lambda mid: contextlib.nullcontext()
    sys.modules["menace.models_repo"] = models_repo

    dyn_mod = types.ModuleType("dynamic_path_router")
    dyn_mod.resolve_path = lambda p: Path(p)
    sys.modules["dynamic_path_router"] = dyn_mod

    ip = importlib.import_module("menace.implementation_pipeline")
    monkeypatch.setattr(
        ip,
        "subprocess",
        types.SimpleNamespace(
            run=lambda *a, **k: types.SimpleNamespace(
                returncode=0, stdout="", stderr=""
            )
        ),
    )
    ip.TaskInfo = TaskInfo
    ip.TaskPackage = TaskPackage
    ip.TaskHandoffBot = StubHandoffBot
    ip.ImplementationOptimiserBot = StubOptimiser
    ip.BotDevelopmentBot = StubDeveloper
    ip.ResearchAggregatorBot = StubResearcher
    ip.IPOBot = StubIPO

    builder = DummyBuilder()
    ipo_instance = StubIPO()
    pipeline = ip.ImplementationPipeline(builder, ipo=ipo_instance)

    task = TaskInfo(
        name="Demo",
        dependencies=[],
        resources={},
        schedule="once",
        code="",
        metadata={"purpose": "demo", "functions": ["run"]},
    )

    pipeline.run([task])

    assert pipeline.context_builder is builder
    assert pipeline.optimiser.context_builder is builder
    assert pipeline.developer.context_builder is builder
    assert pipeline.researcher.context_builder is builder
    assert pipeline.ipo.context_builder is builder
    assert calls == ["Demo"]
