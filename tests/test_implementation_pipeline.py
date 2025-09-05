import os
import types
import sys
import logging
import subprocess
import pytest

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
sys.modules.setdefault("cryptography", types.ModuleType("cryptography"))
sys.modules.setdefault("cryptography.hazmat", types.ModuleType("hazmat"))
sys.modules.setdefault(
    "cryptography.hazmat.primitives", types.ModuleType("primitives")
)
sys.modules.setdefault(
    "cryptography.hazmat.primitives.asymmetric", types.ModuleType("asymmetric")
)
sys.modules.setdefault(
    "cryptography.hazmat.primitives.asymmetric.ed25519",
    types.ModuleType("ed25519"),
)
ed = sys.modules["cryptography.hazmat.primitives.asymmetric.ed25519"]
ed.Ed25519PrivateKey = types.SimpleNamespace(generate=lambda: object())
ed.Ed25519PublicKey = object
serialization = types.ModuleType("serialization")
primitives = sys.modules["cryptography.hazmat.primitives"]
primitives.serialization = serialization
sys.modules.setdefault(
    "cryptography.hazmat.primitives.serialization", serialization
)

sys.modules.setdefault("yaml", types.ModuleType("yaml"))
yaml_mod = sys.modules["yaml"]
yaml_mod.safe_dump = lambda *a, **k: ""
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
sys.modules.setdefault("env_config", types.SimpleNamespace(DATABASE_URL="sqlite:///:memory:"))
sys.modules.setdefault("httpx", types.ModuleType("httpx"))
jinja_mod = types.ModuleType("jinja2")
jinja_mod.Template = lambda *a, **k: None
sys.modules.setdefault("jinja2", jinja_mod)
sys.modules.setdefault("requests", types.ModuleType("requests"))
import menace.implementation_pipeline as ip
import menace.task_handoff_bot as thb
import menace.implementation_optimiser_bot as iob
import menace.bot_development_bot as bdb
import ast
from dataclasses import dataclass
from pathlib import Path


@pytest.fixture(autouse=True)
def _fake_subprocess(monkeypatch):
    def fake_run(*args, **kwargs):
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)


def test_pipeline_runs(tmp_path):
    handoff = thb.TaskHandoffBot()
    optimiser = iob.ImplementationOptimiserBot()
    developer = bdb.BotDevelopmentBot(repo_base=tmp_path)
    pipeline = ip.ImplementationPipeline(
        handoff=handoff,
        optimiser=optimiser,
        developer=developer,
    )

    tasks = [
        thb.TaskInfo(
            name="BotX",
            dependencies=[],
            resources={},
            schedule="once",
            code="print('hi')",
            metadata={
                "purpose": "demo",
                "functions": ["run"],
                "capabilities": ["network"],
                "level": "L2",
                "io_format": "json",
            },
        )
    ]
    result = pipeline.run(tasks)
    assert result.package.tasks[0].name == "BotX"
    plan_str = pipeline._package_to_plan(result.package)
    assert "BotX" in plan_str
    assert "\"capabilities\": [\"network\"]" in plan_str
    assert "\"level\": \"L2\"" in plan_str
    assert "\"io\": \"json\"" in plan_str
    built = tmp_path / "BotX" / "BotX.py"  # path-ignore
    assert built.exists()


def test_prompt_contains_docstrings(tmp_path):
    class CaptureDevBot(bdb.BotDevelopmentBot):
        def __init__(self, repo_base: Path) -> None:  # type: ignore[override]
            super().__init__(repo_base=repo_base)
            self.prompts: list[str] = []

        def build_bot(self, spec: bdb.BotSpec, model_id=None) -> Path:  # type: ignore[override]
            prompt = self._build_prompt(spec)
            self.prompts.append(prompt)
            repo_dir = self.create_env(spec)
            file_path = repo_dir / f"{spec.name}.py"  # path-ignore
            file_path.write_text("pass")
            self._write_meta(repo_dir, spec)
            return file_path

    def _extract(code: str) -> tuple[str, dict[str, str]]:
        tree = ast.parse(code)
        desc = ast.get_docstring(tree) or ""
        docs: dict[str, str] = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node)
                if doc:
                    docs[node.name] = doc
        return desc, docs

    code = '''"""Module doc"""

def run():
    """Run task"""
    pass
'''

    desc, docs = _extract(code)
    handoff = thb.TaskHandoffBot()
    optimiser = iob.ImplementationOptimiserBot()
    developer = CaptureDevBot(repo_base=tmp_path)
    pipeline = ip.ImplementationPipeline(
        handoff=handoff,
        optimiser=optimiser,
        developer=developer,
    )

    tasks = [
        thb.TaskInfo(
            name="DocBot",
            dependencies=[],
            resources={},
            schedule="once",
            code=code,
            metadata={
                "purpose": "demo",
                "functions": ["run"],
                "description": desc,
                "function_docs": docs,
            },
        )
    ]

    pipeline.run(tasks)
    assert any("Module doc" in p for p in developer.prompts)
    assert any("- run: Run task" in p for p in developer.prompts)


def test_prompt_includes_guideline_sections(tmp_path):
    class CaptureDevBot(bdb.BotDevelopmentBot):
        def __init__(self, repo_base: Path) -> None:  # type: ignore[override]
            super().__init__(repo_base=repo_base)
            self.prompt = ""

        def build_bot(self, spec: bdb.BotSpec, model_id=None) -> Path:  # type: ignore[override]
            self.prompt = self._build_prompt(spec)
            repo_dir = self.create_env(spec)
            file_path = repo_dir / f"{spec.name}.py"  # path-ignore
            file_path.write_text("pass")
            self._write_meta(repo_dir, spec)
            return file_path

    developer = CaptureDevBot(repo_base=tmp_path)
    pipeline = ip.ImplementationPipeline(developer=developer)

    tasks = [
        thb.TaskInfo(
            name="GuidBot",
            dependencies=[],
            resources={},
            schedule="once",
            code="",
            metadata={"purpose": "demo", "functions": ["run"]},
        )
    ]

    pipeline.run(tasks)
    prompt = developer.prompt
    assert "INSTRUCTION MODE" in prompt
    assert "Coding Standards:" in prompt
    assert "Repository Layout:" in prompt
    assert "Metadata:" in prompt
    assert "Version Control:" in prompt
    assert "Testing:" in prompt


def test_retry_handoff_and_plan_generation(tmp_path):
    class DummyHandoff(thb.TaskHandoffBot):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[str] = []

        def send_package(self, package: thb.TaskPackage) -> None:  # type: ignore[override]
            self.calls.append("send")
            if len(self.calls) == 2:
                raise RuntimeError("fail")

    class DummyIPO:
        def __init__(self) -> None:
            self.calls = 0

        def generate_plan(self, blueprint: str):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("plan fail")
            return types.SimpleNamespace(
                actions=[types.SimpleNamespace(bot="TaskA", action="demo")],
                graph=types.SimpleNamespace(),
            )

    handoff = DummyHandoff()
    optimiser = iob.ImplementationOptimiserBot()
    developer = bdb.BotDevelopmentBot(repo_base=tmp_path)
    ipo = DummyIPO()
    pipeline = ip.ImplementationPipeline(
        handoff=handoff,
        optimiser=optimiser,
        developer=developer,
        ipo=ipo,
    )

    tasks = [thb.TaskInfo(name="TaskA", dependencies=[], resources={}, schedule="once", code="", metadata={})]
    result = pipeline.run(tasks)
    assert isinstance(result, ip.PipelineResult)
    assert len(handoff.calls) == 3  # initial + two retries
    assert ipo.calls == 2


def test_handoff_fails_after_two_network_errors(tmp_path):
    class FailingHandoff(thb.TaskHandoffBot):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[str] = []

        def send_package(self, package: thb.TaskPackage) -> None:  # type: ignore[override]
            self.calls.append("send")
            if len(self.calls) >= 2:
                raise RuntimeError("network down")

    handoff = FailingHandoff()
    optimiser = iob.ImplementationOptimiserBot()
    developer = bdb.BotDevelopmentBot(repo_base=tmp_path)
    pipeline = ip.ImplementationPipeline(
        handoff=handoff,
        optimiser=optimiser,
        developer=developer,
    )

    tasks = [thb.TaskInfo(name="RetryBot", dependencies=[], resources={}, schedule="once", code="", metadata={})]
    with pytest.raises(RuntimeError):
        pipeline.run(tasks)
    assert len(handoff.calls) == 3  # initial send + two retries


def test_researcher_invoked_for_missing_info(tmp_path):
    class DummyHandoff(thb.TaskHandoffBot):
        def send_package(self, package: thb.TaskPackage) -> None:  # type: ignore[override]
            pass

    class DummyOptimiser:
        def __init__(self) -> None:
            self.calls = 0
            self.from_research = False

        def fill_missing(self, package: thb.TaskPackage, **_kw: str) -> thb.TaskPackage:
            self.calls += 1
            t = package.tasks[0]
            meta = dict(t.metadata or {})
            if self.from_research:
                meta.setdefault("purpose", "demo")
                meta.setdefault("functions", ["run"])
            return thb.TaskPackage(
                tasks=[
                    thb.TaskInfo(
                        name=t.name,
                        dependencies=t.dependencies,
                        resources=t.resources,
                        schedule=t.schedule,
                        code=t.code or "",
                        metadata=meta,
                    )
                ]
            )

        def process(self, package: thb.TaskPackage) -> None:
            pass

    class DummyResearcher:
        def __init__(self, opt: DummyOptimiser) -> None:
            self.called = False
            self.opt = opt

        def process(self, text: str) -> None:
            self.called = True
            self.opt.from_research = True

    class MiniDev(bdb.BotDevelopmentBot):
        def build_bot(self, spec: bdb.BotSpec, model_id=None) -> Path:  # type: ignore[override]
            repo_dir = self.create_env(spec)
            file_path = repo_dir / f"{spec.name}.py"  # path-ignore
            file_path.write_text("pass")
            self._write_meta(repo_dir, spec)
            return file_path

    handoff = DummyHandoff()
    optimiser = DummyOptimiser()
    researcher = DummyResearcher(optimiser)
    developer = MiniDev(repo_base=tmp_path)
    pipeline = ip.ImplementationPipeline(
        handoff=handoff,
        optimiser=optimiser,
        developer=developer,
        researcher=researcher,
        ipo=None,
    )

    tasks = [thb.TaskInfo(name="FillBot", dependencies=[], resources={}, schedule="once", code="", metadata={})]
    result = pipeline.run(tasks)
    assert isinstance(result, ip.PipelineResult)
    assert researcher.called


def test_pipeline_surfaces_openai_errors(tmp_path, monkeypatch, caplog):
    class DummyOpenAI:
        class ChatCompletion:
            @staticmethod
            def create(*a, **k):
                raise RuntimeError("bad")

    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setattr(bdb, "RAISE_ERRORS", True)
    monkeypatch.setattr(bdb, "openai", DummyOpenAI)
    monkeypatch.setattr(bdb, "Repo", None)

    class FallbackDev(bdb.BotDevelopmentBot):
        def __init__(self, repo_base: Path) -> None:  # type: ignore[override]
            super().__init__(repo_base=repo_base, openai_attempts=2)

        def _visual_build(self, prompt: str) -> bool:  # type: ignore[override]
            return False

    developer = FallbackDev(repo_base=tmp_path)
    pipeline = ip.ImplementationPipeline(developer=developer)
    tasks = [
        thb.TaskInfo(
            name="FailBot",
            dependencies=[],
            resources={},
            schedule="once",
            code="print('x')",
            metadata={"purpose": "demo", "functions": ["run"]},
        )
    ]

    caplog.set_level(logging.ERROR)
    with pytest.raises(RuntimeError):
        pipeline.run(tasks)
    assert "openai fallback failed" in caplog.text


def test_pipeline_openai_error_not_raised(tmp_path, monkeypatch, caplog):
    class DummyOpenAI:
        class ChatCompletion:
            @staticmethod
            def create(*a, **k):
                raise RuntimeError("bad")

    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setattr(bdb, "openai", DummyOpenAI)
    monkeypatch.setattr(bdb, "Repo", None)

    class FallbackDev(bdb.BotDevelopmentBot):
        def __init__(self, repo_base: Path) -> None:  # type: ignore[override]
            super().__init__(repo_base=repo_base, openai_attempts=1)

        def _visual_build(self, prompt: str) -> bool:  # type: ignore[override]
            return False

    developer = FallbackDev(repo_base=tmp_path)
    pipeline = ip.ImplementationPipeline(developer=developer)
    tasks = [
        thb.TaskInfo(
            name="FailBot",
            dependencies=[],
            resources={},
            schedule="once",
            code="print('x')",
            metadata={"purpose": "demo", "functions": ["run"]},
        )
    ]

    caplog.set_level(logging.ERROR)
    result = pipeline.run(tasks)
    assert isinstance(result, ip.PipelineResult)
    assert "openai fallback failed" in caplog.text


def test_pipeline_surfaces_build_errors(tmp_path, caplog):
    class FailingDev(bdb.BotDevelopmentBot):
        def build_bot(self, spec: bdb.BotSpec, model_id=None) -> Path:  # type: ignore[override]
            raise RuntimeError("boom")

    developer = FailingDev(repo_base=tmp_path)
    pipeline = ip.ImplementationPipeline(developer=developer)
    tasks = [
        thb.TaskInfo(
            name="ErrBot",
            dependencies=[],
            resources={},
            schedule="once",
            code="",
            metadata={"purpose": "demo", "functions": ["run"]},
        )
    ]

    caplog.set_level(logging.ERROR)
    with pytest.raises(RuntimeError):
        pipeline.run(tasks)
    assert "build_from_plan failed" in caplog.text


def test_ipo_plan_fills_metadata_and_pipeline_runs(tmp_path):
    @dataclass
    class PlanAction:
        bot: str
        action: str

    @dataclass
    class ExecutionPlan:
        actions: list[PlanAction]
        graph: object

    class DummyIPO:
        def generate_plan(self, blueprint: str) -> ExecutionPlan:
            return ExecutionPlan(
                actions=[PlanAction(bot="PlanBot", action="scan")],
                graph=object(),
            )

    class MiniDev(bdb.BotDevelopmentBot):
        def build_bot(self, spec: bdb.BotSpec, model_id=None) -> Path:  # type: ignore[override]
            repo_dir = self.create_env(spec)
            file_path = repo_dir / f"{spec.name}.py"  # path-ignore
            file_path.write_text("pass")
            self._write_meta(repo_dir, spec)
            return file_path

    pipeline = ip.ImplementationPipeline(
        developer=MiniDev(repo_base=tmp_path),
        ipo=DummyIPO(),
    )
    package = thb.TaskPackage(
        tasks=[
            thb.TaskInfo(
                name="PlanBot",
                dependencies=[],
                resources={},
                schedule="once",
                code="",
                metadata={},
            )
        ]
    )

    assert pipeline._apply_ipo_plan(package)
    meta = package.tasks[0].metadata
    assert meta.get("purpose") == "scan"
    assert meta.get("functions") == ["run"]

    result = pipeline.run(package.tasks)
    final_meta = result.package.tasks[0].metadata
    assert final_meta["purpose"] == "scan"
    assert final_meta["functions"] == ["run"]
    built = tmp_path / "PlanBot" / "PlanBot.py"  # path-ignore
    assert built.exists()


def test_pipeline_raises_on_test_failure(tmp_path, monkeypatch):
    def fake_run(cmd, **kwargs):
        if "pytest" in cmd:
            return types.SimpleNamespace(returncode=1, stdout="fail", stderr="err")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    developer = bdb.BotDevelopmentBot(repo_base=tmp_path)
    pipeline = ip.ImplementationPipeline(developer=developer)
    tasks = [
        thb.TaskInfo(
            name="TestBot",
            dependencies=[],
            resources={},
            schedule="once",
            code="",
            metadata={"purpose": "demo", "functions": ["run"]},
        )
    ]

    with pytest.raises(RuntimeError):
        pipeline.run(tasks)
