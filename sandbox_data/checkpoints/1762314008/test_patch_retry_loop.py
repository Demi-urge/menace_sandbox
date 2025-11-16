import sys
import types
from pathlib import Path

import pytest
import contextvars

# Stub out environment_bootstrap if required
stub_env = types.ModuleType("environment_bootstrap")
stub_env.EnvironmentBootstrapper = object
sys.modules.setdefault("environment_bootstrap", stub_env)
db_stub = types.ModuleType("data_bot")
db_stub.MetricsDB = object
sys.modules.setdefault("data_bot", db_stub)
import menace.data_bot as db  # noqa: E402
sys.modules["data_bot"] = db
sys.modules["menace"].RAISE_ERRORS = False
ns = types.ModuleType("neurosales")
ns.add_message = lambda *a, **k: None
ns.get_history = lambda *a, **k: []
ns.get_recent_messages = lambda *a, **k: []
ns.list_conversations = lambda *a, **k: []
sys.modules.setdefault("neurosales", ns)
mapl_stub = types.ModuleType("menace.model_automation_pipeline")


class AutomationResult:
    def __init__(self, package=None, roi=None):
        self.package = package
        self.roi = roi


class ModelAutomationPipeline:
    ...


mapl_stub.AutomationResult = AutomationResult
mapl_stub.ModelAutomationPipeline = ModelAutomationPipeline
sys.modules["menace.model_automation_pipeline"] = mapl_stub
sce_stub = types.ModuleType("menace.self_coding_engine")
sce_stub.SelfCodingEngine = object
sce_stub.MANAGER_CONTEXT = contextvars.ContextVar("MANAGER_CONTEXT")
sys.modules["menace.self_coding_engine"] = sce_stub
import menace.self_coding_manager as scm  # noqa: E402
from menace.model_automation_pipeline import AutomationResult  # noqa: E402


class DummyEngine:
    def __init__(self, builder, patch_db=None):
        self.calls = []
        self.cognition_layer = types.SimpleNamespace(context_builder=builder)
        self.patch_suggestion_db = patch_db

    def apply_patch(self, path: Path, desc: str, **kwargs):
        self.calls.append(kwargs.get("context_meta"))
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("# patched\n")
        return 1, False, 0.0


class DummyPatchDB:
    def __init__(self):
        self.tags = []

    def add_failed_strategy(self, tag):
        self.tags.append(tag)


class DummyPipeline:
    def __init__(self):
        self.calls = []

    def run(self, model: str, energy: int = 1) -> AutomationResult:
        self.calls.append((model, energy))
        return AutomationResult(package=None, roi=None)


def _setup_repo(tmp_path):
    file_path = tmp_path / "sample.py"  # path-ignore
    file_path.write_text("def x():\n    pass\n")
    return file_path


def test_retry_rebuilds_context(monkeypatch, tmp_path):
    query_calls = []

    class DummyContextBuilder:
        def query(self, q, *, exclude_tags=None):
            query_calls.append(exclude_tags)
            return "ctx", "sid"

    builder = DummyContextBuilder()
    patch_db = DummyPatchDB()
    engine = DummyEngine(builder, patch_db)
    pipeline = DummyPipeline()
    mgr = scm.SelfCodingManager(engine, pipeline, bot_name="bot")
    file_path = _setup_repo(tmp_path)
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    tmpdir_path = tmp_path / "clone"

    class DummyTempDir:
        def __enter__(self):
            tmpdir_path.mkdir()
            return str(tmpdir_path)

        def __exit__(self, exc_type, exc, tb):
            for p in tmpdir_path.iterdir():
                p.unlink()
            tmpdir_path.rmdir()

    monkeypatch.setattr(scm.tempfile, "TemporaryDirectory", lambda: DummyTempDir())

    def fake_run(cmd, *a, **kw):
        if cmd[:2] == ["git", "clone"]:
            dst = Path(cmd[3])
            dst.mkdir(exist_ok=True)
            (dst / file_path.name).write_text(file_path.read_text())
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(scm.subprocess, "run", fake_run)

    call_state = {"count": 0}

    def run_tests_stub(repo, path, *, backend="venv"):
        call_state["count"] += 1
        if call_state["count"] == 1:
            return types.SimpleNamespace(
                success=False,
                failure={"strategy_tag": "t1", "stack": "AssertionError: boom"},
                stdout="",
                stderr="",
                duration=0.0,
            )
        return types.SimpleNamespace(
            success=True,
            failure=None,
            stdout="",
            stderr="",
            duration=0.0,
        )

    monkeypatch.setattr(scm, "run_tests", run_tests_stub)

    mgr.run_patch(file_path, "desc")

    assert patch_db.tags == ["t1"]
    assert len(engine.calls) == 2
    assert query_calls == [["t1"]]
    assert pipeline.calls == [("bot", 1)]
    assert call_state["count"] == 2


def test_retry_stops_after_max(monkeypatch, tmp_path):
    class DummyContextBuilder:
        def __init__(self):
            self.calls = []

        def query(self, q, *, exclude_tags=None):
            self.calls.append(exclude_tags)
            return "ctx", "sid"

    builder = DummyContextBuilder()
    engine = DummyEngine(builder)
    pipeline = DummyPipeline()
    mgr = scm.SelfCodingManager(engine, pipeline, bot_name="bot")
    file_path = _setup_repo(tmp_path)
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    tmpdir_path = tmp_path / "clone"

    class DummyTempDir:
        def __enter__(self):
            tmpdir_path.mkdir()
            return str(tmpdir_path)

        def __exit__(self, exc_type, exc, tb):
            for p in tmpdir_path.iterdir():
                p.unlink()
            tmpdir_path.rmdir()

    monkeypatch.setattr(scm.tempfile, "TemporaryDirectory", lambda: DummyTempDir())

    def fake_run(cmd, *a, **kw):
        if cmd[:2] == ["git", "clone"]:
            dst = Path(cmd[3])
            dst.mkdir(exist_ok=True)
            (dst / file_path.name).write_text(file_path.read_text())
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(scm.subprocess, "run", fake_run)

    def run_tests_stub(repo, path, *, backend="venv"):
        return types.SimpleNamespace(
            success=False,
            failure={"strategy_tag": "t1", "stack": "AssertionError: boom"},
            stdout="",
            stderr="",
            duration=0.0,
        )

    monkeypatch.setattr(scm, "run_tests", run_tests_stub)

    with pytest.raises(RuntimeError):
        mgr.run_patch(file_path, "desc", max_attempts=2)

    assert len(engine.calls) == 2
    assert builder.calls == [["t1"]]
    assert not pipeline.calls


def test_retry_skips_duplicate_trace(monkeypatch, tmp_path):
    class DummyContextBuilder:
        def __init__(self):
            self.calls = []

        def query(self, q, *, exclude_tags=None):
            self.calls.append(exclude_tags)
            return "ctx", "sid"

    builder = DummyContextBuilder()
    engine = DummyEngine(builder)
    pipeline = DummyPipeline()
    mgr = scm.SelfCodingManager(engine, pipeline, bot_name="bot")
    file_path = _setup_repo(tmp_path)
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    tmpdir_path = tmp_path / "clone"

    class DummyTempDir:
        def __enter__(self):
            tmpdir_path.mkdir()
            return str(tmpdir_path)

        def __exit__(self, exc_type, exc, tb):
            for p in tmpdir_path.iterdir():
                p.unlink()
            tmpdir_path.rmdir()

    monkeypatch.setattr(scm.tempfile, "TemporaryDirectory", lambda: DummyTempDir())

    def fake_run(cmd, *a, **kw):
        if cmd[:2] == ["git", "clone"]:
            dst = Path(cmd[3])
            dst.mkdir(exist_ok=True)
            (dst / file_path.name).write_text(file_path.read_text())
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(scm.subprocess, "run", fake_run)

    def run_tests_stub(repo, path, *, backend="venv"):
        return types.SimpleNamespace(
            success=False,
            failure={"strategy_tag": "t1", "stack": "AssertionError: boom"},
            stdout="",
            stderr="",
            duration=0.0,
        )

    monkeypatch.setattr(scm, "run_tests", run_tests_stub)

    with pytest.raises(RuntimeError):
        mgr.run_patch(file_path, "desc", max_attempts=3)

    assert len(engine.calls) == 2
    assert builder.calls == [["t1"]]
    assert not pipeline.calls
