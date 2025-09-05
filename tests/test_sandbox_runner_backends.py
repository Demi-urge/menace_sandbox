import importlib
from pathlib import Path
import shutil
import subprocess
import sys
import types
from dynamic_path_router import resolve_path


def _prepare_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "requirements.txt").write_text("")
    tests = repo / "tests"
    tests.mkdir()
    (tests / "test_mod.py").write_text("def test_ok():\n    assert True\n")  # path-ignore
    subprocess.run(["git", "init"], cwd=repo, capture_output=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, capture_output=True)
    return repo


def test_run_tests_supports_backends(monkeypatch, tmp_path):
    repo = _prepare_repo(tmp_path)
    path = resolve_path("sandbox_runner/test_harness.py")  # path-ignore
    pkg = types.ModuleType("menace.sandbox_runner")
    pkg.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "menace.sandbox_runner", pkg)
    spec = importlib.util.spec_from_file_location(
        "menace.sandbox_runner.test_harness", path
    )
    th = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = th
    spec.loader.exec_module(th)
    monkeypatch.setattr(th, "_python_bin", lambda v: Path(sys.executable))

    executed: list[list[str]] = []
    orig_run = subprocess.run

    def fake_run(cmd, *a, cwd=None, capture_output=None, text=None, check=None):
        executed.append(cmd)
        if cmd[:2] == ["git", "clone"]:
            dst = Path(cmd[3])
            shutil.copytree(repo, dst, dirs_exist_ok=True)
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd[0] == sys.executable and cmd[1:3] == ["-m", "venv"]:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd[0] == sys.executable and cmd[1:3] == ["-m", "pip"]:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd[0] == sys.executable and cmd[1:3] == ["-m", "pytest"]:
            return orig_run(cmd, cwd=cwd, capture_output=capture_output, text=text)
        if cmd[0] == "docker":
            idx = cmd.index("-v") + 1
            host_dir = cmd[idx].split(":", 1)[0]
            return orig_run(
                [sys.executable, "-m", "pytest", "-q"],
                cwd=host_dir,
                capture_output=True,
                text=True,
            )
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    res_venv = th.run_tests(repo, backend="venv")
    res_docker = th.run_tests(repo, backend="docker")
    if isinstance(res_venv, list):
        res_venv = res_venv[0]
    if isinstance(res_docker, list):
        res_docker = res_docker[0]

    assert res_venv.success and res_docker.success
    assert res_venv.success == res_docker.success
    assert any(cmd[0] == "docker" for cmd in executed)
    assert any(cmd[0] == sys.executable and cmd[1:3] == ["-m", "venv"] for cmd in executed)


def test_run_patch_forwards_backend(monkeypatch, tmp_path):
    import types, sys

    stub_env = types.ModuleType("environment_bootstrap")
    stub_env.EnvironmentBootstrapper = object
    monkeypatch.setitem(sys.modules, "environment_bootstrap", stub_env)
    db_stub = types.ModuleType("data_bot")
    db_stub.MetricsDB = object
    monkeypatch.setitem(sys.modules, "data_bot", db_stub)
    import menace.data_bot as db
    monkeypatch.setitem(sys.modules, "data_bot", db)
    ns = types.ModuleType("neurosales")
    ns.add_message = lambda *a, **k: None
    ns.get_history = lambda *a, **k: []
    ns.get_recent_messages = lambda *a, **k: []
    ns.list_conversations = lambda *a, **k: []
    monkeypatch.setitem(sys.modules, "neurosales", ns)
    mapl_stub = types.ModuleType("menace.model_automation_pipeline")
    class AutomationResult:
        def __init__(self, package=None, roi=None):
            self.package = package
            self.roi = roi
    class ModelAutomationPipeline: ...
    mapl_stub.AutomationResult = AutomationResult
    mapl_stub.ModelAutomationPipeline = ModelAutomationPipeline
    monkeypatch.setitem(sys.modules, "menace.model_automation_pipeline", mapl_stub)
    sce_stub = types.ModuleType("menace.self_coding_engine")
    sce_stub.SelfCodingEngine = object
    monkeypatch.setitem(sys.modules, "menace.self_coding_engine", sce_stub)
    prb_stub = types.ModuleType("menace.pre_execution_roi_bot")
    class ROIResult:
        def __init__(self, roi, errors, proi, perr, risk):
            self.roi = roi
            self.errors = errors
            self.predicted_roi = proi
            self.predicted_errors = perr
            self.risk = risk
    prb_stub.ROIResult = ROIResult
    monkeypatch.setitem(sys.modules, "menace.pre_execution_roi_bot", prb_stub)
    import menace.self_coding_manager as scm

    file_path = tmp_path / "sample.py"  # path-ignore
    file_path.write_text("x = 1\n")

    class Engine:
        def apply_patch(self, path, desc, **kw):
            return 1, False, 0.0

    class Pipeline:
        def run(self, model, energy=1):
            return None

    mgr = scm.SelfCodingManager(Engine(), Pipeline(), bot_name="bot")
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    tmpdir_path = tmp_path / "clone"

    class DummyTempDir:
        def __enter__(self):
            tmpdir_path.mkdir()
            return str(tmpdir_path)

        def __exit__(self, exc_type, exc, tb):
            shutil.rmtree(tmpdir_path)

    monkeypatch.setattr(scm.tempfile, "TemporaryDirectory", lambda: DummyTempDir())

    def fake_run(cmd, *a, **kw):
        if cmd[:2] == ["git", "clone"]:
            dst = Path(cmd[3])
            dst.mkdir(exist_ok=True)
            shutil.copy2(file_path, dst / file_path.name)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(scm.subprocess, "run", fake_run)

    called = {}

    def run_tests_stub(repo, path, *, backend="venv"):
        called["backend"] = backend
        return types.SimpleNamespace(
            success=True,
            failure=None,
            stdout="",
            stderr="",
            duration=0.0,
        )

    monkeypatch.setattr(scm, "run_tests", run_tests_stub)

    mgr.run_patch(file_path, "add", backend="docker")
    assert called["backend"] == "docker"
