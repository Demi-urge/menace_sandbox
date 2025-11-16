import math
import sys
import types
from pathlib import Path

import pytest

db_stub = types.ModuleType("data_bot")
db_stub.MetricsDB = object
sys.modules.setdefault("data_bot", db_stub)
import menace.data_bot as db  # noqa: E402
sys.modules["data_bot"] = db

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
sys.modules["menace.self_coding_engine"] = sce_stub

import menace.self_coding_manager as scm  # noqa: E402
from menace.failure_fingerprint_store import FailureFingerprint


TRACE = (
    "Traceback (most recent call last):\n"
    '  File "mod.py", line 1, in <module>\n'  # path-ignore
    "    1/0\n"
    "ZeroDivisionError: division by zero"
)


class DummyStore:
    def __init__(self, sim: float | None):
        self.sim = sim

    def find_similar(self, fp: FailureFingerprint):
        fp.embedding = [1.0, 0.0]
        if self.sim is None:
            return []
        emb = [self.sim, math.sqrt(1 - self.sim**2)]
        match = FailureFingerprint("a.py", "f", "e", "t", "p", embedding=emb)  # path-ignore
        return [match]


class DummyContextBuilder:
    def query(self, q, *, exclude_tags=None):
        return "ctx", "sid"


class DummyEngine:
    def __init__(self, builder):
        self.calls = []
        self.cognition_layer = types.SimpleNamespace(context_builder=builder)

    def apply_patch(self, path: Path, desc: str, **kwargs):
        self.calls.append(desc)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("# patched\n")
        return 1, False, 0.0


class DummyPipeline:
    def run(self, model: str, energy: int = 1):
        return types.SimpleNamespace(package=None, roi=None)


def _setup_repo(tmp_path: Path) -> Path:
    file_path = tmp_path / "sample.py"  # path-ignore
    file_path.write_text("def x():\n    pass\n")
    return file_path


def _patch_env(monkeypatch, tmp_path: Path, file_path: Path):
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


def test_fingerprint_warning(monkeypatch, tmp_path):
    builder = DummyContextBuilder()
    engine = DummyEngine(builder)
    pipeline = DummyPipeline()
    store = DummyStore(0.9)
    mgr = scm.SelfCodingManager(
        engine, pipeline, bot_name="bot", failure_store=store, skip_similarity=0.95
    )
    file_path = _setup_repo(tmp_path)
    _patch_env(monkeypatch, tmp_path, file_path)

    call_state = {"count": 0}

    def run_tests_stub(repo, path, *, backend="venv"):
        call_state["count"] += 1
        if call_state["count"] == 1:
            return types.SimpleNamespace(
                success=True, failure=None, stdout="", stderr="", duration=0.0
            )
        if call_state["count"] == 2:
            return types.SimpleNamespace(
                success=False,
                failure={"strategy_tag": "t1", "stack": TRACE},
                stdout="",
                stderr="",
                duration=0.0,
            )
        return types.SimpleNamespace(
            success=True, failure=None, stdout="", stderr="", duration=0.0
        )

    monkeypatch.setattr(scm, "run_tests", run_tests_stub)

    mgr.run_patch(file_path, "desc")
    assert len(engine.calls) == 2
    assert "ZeroDivisionError: division by zero" in engine.calls[1]


def test_fingerprint_skip(monkeypatch, tmp_path):
    builder = DummyContextBuilder()
    engine = DummyEngine(builder)
    pipeline = DummyPipeline()
    store = DummyStore(0.99)
    mgr = scm.SelfCodingManager(
        engine, pipeline, bot_name="bot", failure_store=store, skip_similarity=0.95
    )
    file_path = _setup_repo(tmp_path)
    _patch_env(monkeypatch, tmp_path, file_path)

    def run_tests_stub(repo, path, *, backend="venv"):
        if not hasattr(run_tests_stub, "called"):
            run_tests_stub.called = False  # type: ignore[attr-defined]
            return types.SimpleNamespace(
                success=True, failure=None, stdout="", stderr="", duration=0.0
            )
        return types.SimpleNamespace(
            success=False,
            failure={"strategy_tag": "t1", "stack": TRACE},
            stdout="",
            stderr="",
            duration=0.0,
        )

    monkeypatch.setattr(scm, "run_tests", run_tests_stub)

    with pytest.raises(RuntimeError, match="similar failure detected"):
        mgr.run_patch(file_path, "desc", max_attempts=2)
    assert len(engine.calls) == 1

