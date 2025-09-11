import os
import types
import sys
from pathlib import Path
import subprocess
import shutil
import pytest

# Stub out environment_bootstrap if required
stub_env = types.ModuleType("environment_bootstrap")
stub_env.EnvironmentBootstrapper = object
sys.modules.setdefault("environment_bootstrap", stub_env)

# Data bot shim
db_stub = types.ModuleType("data_bot")
db_stub.MetricsDB = object
sys.modules.setdefault("data_bot", db_stub)
import menace.data_bot as db
sys.modules["data_bot"] = db

# Misc optional deps
sys.modules["menace"].RAISE_ERRORS = False
ns = types.ModuleType("neurosales")
ns.add_message = lambda *a, **k: None
ns.get_history = lambda *a, **k: []
ns.get_recent_messages = lambda *a, **k: []
ns.list_conversations = lambda *a, **k: []
ns.push_chain = lambda *a, **k: None
ns.peek_chain = lambda *a, **k: None
sys.modules.setdefault("neurosales", ns)

# Model automation pipeline / ROI stubs
mapl_stub = types.ModuleType("menace.model_automation_pipeline")
class AutomationResult:
    def __init__(self, package=None, roi=None):
        self.package = package
        self.roi = roi
class ModelAutomationPipeline: ...
mapl_stub.AutomationResult = AutomationResult
mapl_stub.ModelAutomationPipeline = ModelAutomationPipeline
sys.modules["menace.model_automation_pipeline"] = mapl_stub

prb_stub = types.ModuleType("menace.pre_execution_roi_bot")
class ROIResult:
    def __init__(self, roi, confidence=1.0, errors=0.0, proi=0.0, perr=0.0, risk=0.0):
        self.roi = roi
        self.confidence = confidence
        self.errors = errors
        self.predicted_roi = proi
        self.predicted_errors = perr
        self.risk = risk
prb_stub.ROIResult = ROIResult
sys.modules["menace.pre_execution_roi_bot"] = prb_stub

# Sandbox harness stub to avoid importing heavy dependencies
sandbox_stub = types.ModuleType("sandbox_runner.test_harness")
sandbox_stub.run_tests = lambda *a, **k: None
sandbox_stub.TestHarnessResult = object
sys.modules["sandbox_runner.test_harness"] = sandbox_stub

sce_stub = types.ModuleType("menace.self_coding_engine")
sce_stub.SelfCodingEngine = object
sys.modules["menace.self_coding_engine"] = sce_stub

# Patch provenance stub
pp_stub = types.ModuleType("menace.patch_provenance")
pp_stub.record_patch_metadata = lambda *a, **k: None
sys.modules["menace.patch_provenance"] = pp_stub

import menace.self_coding_manager as scm


class DummyPatchLogger:
    def __init__(self):
        self.calls = []
    def track_contributors(self, *a, **k):
        self.calls.append((a, k))


class DummyContextBuilder:
    def __init__(self):
        self.calls = []
    def query(self, q, *, exclude_tags=None):
        self.calls.append(exclude_tags)
        return "ctx", "sid"


class DummyEngine:
    def __init__(self, builder, logger):
        self.cognition_layer = types.SimpleNamespace(context_builder=builder)
        self.patch_logger = logger
        self.calls = []
    def apply_patch(self, path: Path, desc: str, **kwargs):
        self.calls.append(kwargs.get("context_meta"))
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("# patched\n")
        return 1, False, 0.0


class DummyPipeline:
    def run(self, model: str, energy: int = 1) -> AutomationResult:
        roi = prb_stub.ROIResult(roi=1.0, confidence=0.9)
        return AutomationResult(package=None, roi=roi)


class DummyDataBot:
    def __init__(self):
        self._vals = [0.0, 1.0]
        self.logged = []
    def roi(self, _name):
        return self._vals.pop(0)
    def log_evolution_cycle(self, *a, **k):
        self.logged.append(k)
    def average_errors(self, _name):  # pragma: no cover - simple
        return 0.0


@pytest.mark.parametrize("backend", ["venv", "docker"])
def test_closed_loop_patch(monkeypatch, tmp_path, backend):
    file_path = tmp_path / "sample.py"  # path-ignore
    file_path.write_text("def x():\n    return 1\n", encoding="utf-8")
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    pushes: list[list[str]] = []

    def fake_run(cmd, *a, cwd=None, check=None, **kw):
        if cmd[:2] == ["git", "clone"]:
            dst = Path(cmd[3])
            dst.mkdir(exist_ok=True)
            shutil.copy2(file_path, dst / file_path.name)
            return subprocess.CompletedProcess(cmd, 0)
        if cmd[:2] == ["git", "push"]:
            pushes.append(cmd)
            return subprocess.CompletedProcess(cmd, 0)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(scm.subprocess, "run", fake_run)

    tmpdir_path = tmp_path / "clone"
    class DummyTempDir:
        def __enter__(self):
            tmpdir_path.mkdir()
            return str(tmpdir_path)
        def __exit__(self, exc_type, exc, tb):
            shutil.rmtree(tmpdir_path)
    monkeypatch.setattr(scm.tempfile, "TemporaryDirectory", lambda: DummyTempDir())

    run_calls: list[str] = []

    def run_tests_stub(repo, path, *, backend="venv"):
        run_calls.append("fail" if not run_calls else "pass")
        if len(run_calls) == 1:
            return types.SimpleNamespace(
                success=False,
                failure={"strategy_tag": "boom", "stack": "Traceback"},
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
    monkeypatch.setattr(scm.MutationLogger, "log_mutation", lambda *a, **k: 1)
    monkeypatch.setattr(scm.MutationLogger, "record_mutation_outcome", lambda *a, **k: None)

    builder = DummyContextBuilder()
    logger = DummyPatchLogger()
    engine = DummyEngine(builder, logger)
    pipeline = DummyPipeline()
    data_bot = DummyDataBot()
    mgr = scm.SelfCodingManager(engine, pipeline, bot_name="bot", data_bot=data_bot)

    mgr.run_patch(file_path, "change")

    # Failure tags should be forwarded to the context builder only once
    assert builder.calls == [["boom"]]
    assert run_calls == ["fail", "pass"]
    # Patch commit and ROI tracking side effects
    assert pushes and pushes[-1][-1].startswith("review/")
    assert data_bot.logged and data_bot.logged[0]["roi_delta"] == 1.0
    assert logger.calls and logger.calls[0][1]["contribution"] == 1.0
    # Source file should have been modified by the dummy patch
    assert "# patched" in file_path.read_text(encoding="utf-8")


def test_run_patch_failure_no_attribute_error(monkeypatch, tmp_path):
    file_path = tmp_path / "sample.py"  # path-ignore
    file_path.write_text("def x():\n    return 1\n", encoding="utf-8")
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    def fake_run(cmd, *a, cwd=None, check=None, **kw):
        if cmd[:2] == ["git", "clone"]:
            dst = Path(cmd[3])
            dst.mkdir(exist_ok=True)
            shutil.copy2(file_path, dst / file_path.name)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(scm.subprocess, "run", fake_run)

    tmpdir_path = tmp_path / "clone"

    class DummyTempDir:
        def __enter__(self):
            tmpdir_path.mkdir()
            return str(tmpdir_path)

        def __exit__(self, exc_type, exc, tb):
            shutil.rmtree(tmpdir_path)

    monkeypatch.setattr(scm.tempfile, "TemporaryDirectory", lambda: DummyTempDir())

    def run_tests_stub(repo, path, *, backend="venv"):
        return types.SimpleNamespace(
            success=False,
            failure=None,
            stdout="",
            stderr="AssertionError: boom",
            duration=0.0,
        )

    monkeypatch.setattr(scm, "run_tests", run_tests_stub)

    builder = DummyContextBuilder()
    logger = DummyPatchLogger()
    engine = DummyEngine(builder, logger)
    pipeline = DummyPipeline()
    data_bot = DummyDataBot()
    mgr = scm.SelfCodingManager(engine, pipeline, bot_name="bot", data_bot=data_bot)

    with pytest.raises(RuntimeError):
        mgr.run_patch(file_path, "change", max_attempts=2)


def _prepare_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "mod.py").write_text("def mod():\n    return 1\n", encoding="utf-8")  # path-ignore
    tests = repo / "tests"
    tests.mkdir()
    (tests / "test_mod.py").write_text(  # path-ignore
        "from mod import mod\n\n" "def test_mod():\n    assert mod() == 1\n",
        encoding="utf-8",
    )
    (tests / "test_other.py").write_text(  # path-ignore
        "def test_other():\n    assert True\n", encoding="utf-8"
    )
    return repo, tests


def test_harness_filters_specific_test_file(monkeypatch, tmp_path):
    import importlib

    repo, tests = _prepare_repo(tmp_path)
    th = importlib.import_module("menace.sandbox_runner.test_harness")
    monkeypatch.setattr(th, "_python_bin", lambda v: Path(sys.executable))

    pytest_cmds = []

    def fake_run(cmd, *a, cwd=None, capture_output=None, text=None, check=None):
        if cmd[:2] == ["git", "clone"]:
            dst = Path(cmd[3])
            shutil.copytree(repo, dst, dirs_exist_ok=True)
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd[0] == sys.executable and cmd[1:3] == ["-m", "venv"]:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd[0] == sys.executable and cmd[1:3] == ["-m", "pip"]:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd[0] == sys.executable and cmd[1:3] == ["-m", "pytest"]:
            pytest_cmds.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = th.run_tests(repo, tests / "test_mod.py")  # path-ignore
    if isinstance(result, list):
        result = result[0]

    assert pytest_cmds and str(Path("tests/test_mod.py")) in pytest_cmds[0]  # path-ignore
    assert "-k" not in pytest_cmds[0]
    assert result.path == "tests/test_mod.py"  # path-ignore


def test_harness_uses_k_filter_for_module(monkeypatch, tmp_path):
    import importlib

    repo, tests = _prepare_repo(tmp_path)
    th = importlib.import_module("menace.sandbox_runner.test_harness")
    monkeypatch.setattr(th, "_python_bin", lambda v: Path(sys.executable))

    pytest_cmds = []

    def fake_run(cmd, *a, cwd=None, capture_output=None, text=None, check=None):
        if cmd[:2] == ["git", "clone"]:
            dst = Path(cmd[3])
            shutil.copytree(repo, dst, dirs_exist_ok=True)
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd[0] == sys.executable and cmd[1:3] == ["-m", "venv"]:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd[0] == sys.executable and cmd[1:3] == ["-m", "pip"]:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd[0] == sys.executable and cmd[1:3] == ["-m", "pytest"]:
            pytest_cmds.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = th.run_tests(repo, repo / "mod.py")  # path-ignore
    if isinstance(result, list):
        result = result[0]

    assert pytest_cmds and "-k" in pytest_cmds[0]
    k_index = pytest_cmds[0].index("-k") + 1
    assert pytest_cmds[0][k_index] == "mod"
    assert result.path == "mod.py"  # path-ignore
