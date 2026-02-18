# flake8: noqa
import pytest

pytest.importorskip("networkx")
pytest.importorskip("pandas")

import sys
import json
import types
stub_env = types.ModuleType("environment_bootstrap")
stub_env.EnvironmentBootstrapper = object
sys.modules.setdefault("environment_bootstrap", stub_env)
db_stub = types.ModuleType("data_bot")
db_stub.MetricsDB = object
sys.modules.setdefault("data_bot", db_stub)
db_router_stub = types.ModuleType("db_router")
db_router_stub.GLOBAL_ROUTER = None
db_router_stub.LOCAL_TABLES = set()


class DummyRouter:
    menace_id = "test"

    def __init__(self, *a, **k):
        pass

    class _Conn:
        def execute(self, *a, **k):
            return types.SimpleNamespace(fetchall=lambda: [])

        def commit(self):
            pass

    def get_connection(self, *_a, **_k):
        return self._Conn()


def init_db_router(*a, **k):
    return DummyRouter()


db_router_stub.DBRouter = DummyRouter
db_router_stub.init_db_router = init_db_router
sys.modules.setdefault("db_router", db_router_stub)
dpr = types.SimpleNamespace(
    resolve_path=lambda p: __import__("pathlib").Path(p),
    repo_root=lambda: __import__("pathlib").Path("."),
    path_for_prompt=lambda p: str(p),
    get_project_root=lambda: __import__("pathlib").Path("."),
)
sys.modules["dynamic_path_router"] = dpr
import menace.data_bot as db
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
class ModelAutomationPipeline: ...
mapl_stub.AutomationResult = AutomationResult
mapl_stub.ModelAutomationPipeline = ModelAutomationPipeline
sys.modules["menace.model_automation_pipeline"] = mapl_stub
sce_stub = types.ModuleType("menace.self_coding_engine")
sce_stub.SelfCodingEngine = object
sce_stub.MANAGER_CONTEXT = {}
sys.modules["menace.self_coding_engine"] = sce_stub
cb_stub = types.ModuleType("coding_bot_interface")
cb_stub.manager_generate_helper = lambda *a, **k: ""
sys.modules.setdefault("coding_bot_interface", cb_stub)
qfe_stub = types.ModuleType("quick_fix_engine")

class QuickFixEngineError(Exception):
    pass

qfe_stub.QuickFixEngine = object
qfe_stub.QuickFixEngineError = QuickFixEngineError
qfe_stub.generate_patch = lambda *a, **k: (1, [])
sys.modules.setdefault("quick_fix_engine", qfe_stub)
sys.modules.setdefault("menace.quick_fix_engine", qfe_stub)
prb_stub = types.ModuleType("menace.pre_execution_roi_bot")
class ROIResult:
    def __init__(self, roi, errors, proi, perr, risk):
        self.roi = roi
        self.errors = errors
        self.predicted_roi = proi
        self.predicted_errors = perr
        self.risk = risk
prb_stub.ROIResult = ROIResult
sys.modules["menace.pre_execution_roi_bot"] = prb_stub
error_bot_stub = types.ModuleType("menace.error_bot")
error_bot_stub.ErrorDB = object
sys.modules.setdefault("menace.error_bot", error_bot_stub)
aem_stub = types.ModuleType("menace.advanced_error_management")
aem_stub.FormalVerifier = object
aem_stub.AutomatedRollbackManager = object
sys.modules.setdefault("menace.advanced_error_management", aem_stub)
rm_stub = types.ModuleType("menace.rollback_manager")
rm_stub.RollbackManager = object
sys.modules.setdefault("menace.rollback_manager", rm_stub)
mutation_logger_stub = types.ModuleType("menace.mutation_logger")
mutation_logger_stub.log_mutation = lambda *a, **k: None
sys.modules.setdefault("menace.mutation_logger", mutation_logger_stub)
sr_pkg = types.ModuleType("menace.sandbox_runner")
th_stub = types.ModuleType("menace.sandbox_runner.test_harness")
th_stub.run_tests = lambda *a, **k: types.SimpleNamespace(
    success=True, failure=None, stdout="", stderr="", duration=0.0
)
th_stub.TestHarnessResult = types.SimpleNamespace
sr_pkg.test_harness = th_stub
sys.modules.setdefault("menace.sandbox_runner", sr_pkg)
sys.modules.setdefault("menace.sandbox_runner.test_harness", th_stub)
code_db_stub = types.ModuleType("menace.code_database")
class PatchRecord:
    pass
class PatchHistoryDB:
    pass
code_db_stub.PatchRecord = PatchRecord
code_db_stub.PatchHistoryDB = PatchHistoryDB
class CodeDB:
    pass
code_db_stub.CodeDB = CodeDB
sys.modules["menace.code_database"] = code_db_stub
sys.modules["code_database"] = code_db_stub
import menace.self_coding_manager as scm
from sandbox_settings import normalize_workflow_tests as real_normalize
import menace.model_automation_pipeline as mapl
import menace.pre_execution_roi_bot as prb
from menace.evolution_history_db import EvolutionHistoryDB
from pathlib import Path
import subprocess
import tempfile
import shutil
import logging
import sqlite3


class DummyEngine:
    def __init__(self):
        self.calls = []

    def apply_patch(self, path: Path, desc: str, **_: object):
        self.calls.append((path, desc))
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("# patched\n")
        return 1, False, 0.0


class DummyPipeline:
    def __init__(self):
        self.calls = []

    def run(self, model: str, energy: int = 1) -> mapl.AutomationResult:
        self.calls.append((model, energy))
        return mapl.AutomationResult(
            package=None,
            roi=prb.ROIResult(1.0, 0.5, 1.0, 0.5, 0.1),
        )


class DummyRegistry:
    class Graph:
        def __init__(self) -> None:
            self.nodes: dict[str, dict] = {}

        def __contains__(self, name: str) -> bool:
            return name in self.nodes

    def __init__(self) -> None:
        self.graph = self.Graph()

    def register_bot(self, name: str, **_k) -> None:
        pass

    def record_validation(self, *a, **k) -> None:
        pass

    def record_heartbeat(self, *a, **k) -> None:
        pass

    def register_interaction(self, *a, **k) -> None:
        pass

    def record_interaction_metadata(self, *a, **k) -> None:
        pass

    def update_bot(self, *a, **k) -> None:
        pass

    def hot_swap_bot(self, *a, **k) -> None:
        pass


class DummyDataBot:
    def __init__(self) -> None:
        self.failures = 0
        self.db = types.SimpleNamespace(log_eval=lambda *a, **k: None)

    def roi(self, _bot: str) -> float:
        return 1.0

    def average_errors(self, _bot: str) -> float:
        return 0.0

    def average_test_failures(self, _bot: str) -> float:
        return self.failures

    def get_thresholds(self, _bot: str):
        return types.SimpleNamespace(
            roi_drop=-999.0, error_threshold=999.0, test_failure_threshold=1.0
        )

    def log_evolution_cycle(self, *a, **k) -> None:  # pragma: no cover - simple
        pass


def _make_manager(bot_name: str = "bot") -> scm.SelfCodingManager:
    manager = object.__new__(scm.SelfCodingManager)
    manager.pipeline = types.SimpleNamespace(
        workflow_test_args=None,
        workflow_test_workers=None,
        workflow_test_kwargs=None,
    )
    manager.engine = types.SimpleNamespace(workflow_test_args=None)
    manager.data_bot = types.SimpleNamespace(workflow_test_args=None)
    manager.bot_name = bot_name
    manager.logger = logging.getLogger(f"workflow-test-{bot_name}")

    class _Graph:
        def __init__(self) -> None:
            self.nodes: dict[str, dict[str, object]] = {}

        def __contains__(self, name: str) -> bool:
            return name in self.nodes

    manager.bot_registry = types.SimpleNamespace(
        graph=_Graph(),
        modules={},
        _lock=None,
        register_bot=lambda *a, **k: None,
    )
    return manager


def test_workflow_args_registry_fallback(monkeypatch):
    manager = _make_manager("registry_bot")
    manager.bot_registry.graph.nodes["registry_bot"] = {
        "workflow_tests": ["tests/test_registry_path.py"],
    }
    monkeypatch.setattr(
        scm,
        "get_bot_workflow_tests",
        lambda name, registry=None, **_: list(
            registry.graph.nodes.get(name, {}).get("workflow_tests", [])
        ),
    )
    pytest_args, kwargs, tests, sources = manager._workflow_test_service_args()
    assert pytest_args == "tests/test_registry_path.py"
    assert kwargs == {}
    assert tests == ["tests/test_registry_path.py"]
    assert sources == {"registry": ["tests/test_registry_path.py"]}


def test_workflow_args_summary_fallback(tmp_path, monkeypatch):
    manager = _make_manager("summary_bot")
    manager._historical_workflow_tests = ["tests/test_summary_selection.py"]
    normalize_calls: list[tuple[object, list[str]]] = []
    original_normalize = real_normalize

    def _track_normalize(value):
        result = original_normalize(value)
        normalize_calls.append((value, list(result)))
        return result

    monkeypatch.setattr(scm, "normalize_workflow_tests", _track_normalize)
    monkeypatch.setattr(scm, "resolve_path", lambda value: tmp_path / value)
    monkeypatch.setattr(
        scm,
        "get_bot_workflow_tests",
        lambda *_a, **_k: [],
    )
    try:
        pytest_args, kwargs, tests, sources = manager._workflow_test_service_args()
    except RuntimeError as exc:  # pragma: no cover - debug aid
        pytest.fail(f"summary fallback failed: {exc}; normalize_calls={normalize_calls}")
    assert pytest_args == "tests/test_summary_selection.py"
    assert kwargs == {}
    assert tests == ["tests/test_summary_selection.py"]
    assert sources == {"summary": ["tests/test_summary_selection.py"]}


def test_workflow_args_heuristic_fallback(tmp_path, monkeypatch):
    manager = _make_manager("alpha_bot")
    project_root = tmp_path
    tests_dir = project_root / "tests"
    tests_dir.mkdir()
    test_file = tests_dir / "test_alpha_bot.py"
    test_file.write_text("def test_placeholder():\n    assert True\n")
    module_dir = project_root / "src"
    module_dir.mkdir()
    module_path = module_dir / "alpha_bot.py"
    module_path.write_text("value = 1\n")
    manager.bot_registry.graph.nodes["alpha_bot"] = {
        "module": str(module_path),
    }
    monkeypatch.chdir(project_root)
    monkeypatch.setattr(scm, "resolve_path", lambda value: project_root / value)
    monkeypatch.setattr(
        scm,
        "get_bot_workflow_tests",
        lambda *_a, **_k: [],
    )
    pytest_args, kwargs, tests, sources = manager._workflow_test_service_args()
    expected_path = str(test_file.resolve())
    assert pytest_args == expected_path
    assert kwargs == {}
    assert tests == [expected_path]
    assert sources == {"heuristic": [expected_path]}


def test_workflow_args_failure_when_no_tests(tmp_path, monkeypatch):
    manager = _make_manager("ghost_bot")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(scm, "resolve_path", lambda value: tmp_path / value)
    monkeypatch.setattr(
        scm,
        "get_bot_workflow_tests",
        lambda *_a, **_k: [],
    )
    pytest_args, kwargs, tests, sources = manager._workflow_test_service_args()
    assert pytest_args is None
    assert kwargs == {}
    assert tests == []
    assert sources == {}


def test_run_patch_logs_evolution(monkeypatch, tmp_path):
    hist = EvolutionHistoryDB(tmp_path / "e.db")

    class LocalDataBot(DummyDataBot):
        def log_evolution_cycle(self, *a, **k) -> None:
            hist.log_cycle("self_coding", {})

    data_bot = LocalDataBot()
    engine = DummyEngine()
    pipeline = DummyPipeline()
    mgr = scm.SelfCodingManager(
        engine,
        pipeline,
        bot_name="bot",
        data_bot=data_bot,
        bot_registry=DummyRegistry(),
    )
    file_path = tmp_path / "sample.py"  # path-ignore
    file_path.write_text("def x():\n    pass\n")
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    tmpdir_path = tmp_path / "clone"

    class DummyTempDir:
        def __enter__(self):
            tmpdir_path.mkdir()
            return str(tmpdir_path)

        def __exit__(self, exc_type, exc, tb):
            shutil.rmtree(tmpdir_path)

    monkeypatch.setattr(tempfile, "TemporaryDirectory", lambda: DummyTempDir())

    def fake_run(cmd, *a, **kw):
        if cmd[:2] == ["git", "clone"]:
            dst = Path(cmd[3])
            dst.mkdir(exist_ok=True)
            shutil.copy2(file_path, dst / file_path.name)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(scm.subprocess, "run", fake_run)

    calls: list[tuple] = []

    def run_tests_stub(repo, path, *, backend="venv"):
        calls.append((repo, path, backend))
        return types.SimpleNamespace(
            success=True,
            failure=None,
            stdout="",
            stderr="",
            duration=0.0,
        )

    monkeypatch.setattr(scm, "run_tests", run_tests_stub)

    res = mgr.run_patch(file_path, "add")
    assert engine.calls
    assert pipeline.calls
    assert calls
    assert "# patched" in file_path.read_text()
    rows = hist.fetch()
    assert any(r[0].startswith("self_coding") for r in rows)
    assert isinstance(res, mapl.AutomationResult)


def test_run_patch_logging_error(monkeypatch, tmp_path, caplog):
    hist = EvolutionHistoryDB(tmp_path / "e.db")

    class LocalDataBot(DummyDataBot):
        def log_evolution_cycle(self, *a, **k) -> None:
            hist.log_cycle("self_coding", {})

    data_bot = LocalDataBot()
    engine = DummyEngine()
    pipeline = DummyPipeline()
    mgr = scm.SelfCodingManager(
        engine,
        pipeline,
        bot_name="bot",
        data_bot=data_bot,
        bot_registry=DummyRegistry(),
    )
    file_path = tmp_path / "sample.py"  # path-ignore
    file_path.write_text("def x():\n    pass\n")
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    tmpdir_path = tmp_path / "clone"

    class DummyTempDir:
        def __enter__(self):
            tmpdir_path.mkdir()
            return str(tmpdir_path)

        def __exit__(self, exc_type, exc, tb):
            shutil.rmtree(tmpdir_path)

    monkeypatch.setattr(tempfile, "TemporaryDirectory", lambda: DummyTempDir())

    def fake_run(cmd, *a, **kw):
        if cmd[:2] == ["git", "clone"]:
            dst = Path(cmd[3])
            dst.mkdir(exist_ok=True)
            shutil.copy2(file_path, dst / file_path.name)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(scm.subprocess, "run", fake_run)

    monkeypatch.setattr(
        scm,
        "run_tests",
        lambda repo, path: types.SimpleNamespace(
            success=True,
            failure=None,
            stdout="",
            stderr="",
            duration=0.0,
        ),
    )

    def fail(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(data_bot, "log_evolution_cycle", fail)
    caplog.set_level(logging.ERROR)
    mgr.run_patch(file_path, "add")
    assert "failed to log evolution cycle" in caplog.text


def test_approval_logs_audit_failure(monkeypatch, tmp_path, caplog):
    class DummyVerifier:
        def verify(self, path: Path) -> bool:
            return True

    class DummyRollback:
        def log_healing_action(self, *a, **k):
            raise RuntimeError("boom")

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 0),
    )
    policy = scm.PatchApprovalPolicy(
        verifier=DummyVerifier(),
        rollback_mgr=DummyRollback(),
        bot_name="bot",
    )
    caplog.set_level(logging.ERROR)
    file_path = tmp_path / "x.py"  # path-ignore
    file_path.write_text("x = 1\n")
    assert policy.approve(file_path)
    assert "failed to log healing action" in caplog.text


def test_run_patch_records_patch_outcome(monkeypatch, tmp_path):
    builder = types.SimpleNamespace()

    class DummyEngine:
        def __init__(self):
            self.cognition_layer = types.SimpleNamespace(
                calls=[], context_builder=builder
            )

        def apply_patch(self, path: Path, desc: str, **_: object):
            with open(path, "a", encoding="utf-8") as fh:
                fh.write("# patched\n")
            return 1, False, 0.0

    class DummyPipeline:
        def run(self, model: str, energy: int = 1) -> mapl.AutomationResult:
            return mapl.AutomationResult(package=None, roi=None)

    class DummyDataBot:
        def __init__(self):
            self._vals = iter([1.0, 2.0])

        def roi(self, _bot: str) -> float:
            return next(self._vals)

        def log_evolution_cycle(self, *a, **k):
            pass
        def average_errors(self, _bot: str) -> float:  # pragma: no cover - simple
            return 0.0

        def average_test_failures(self, _bot: str) -> float:  # pragma: no cover - simple
            return 0.0

        def get_thresholds(self, _bot: str):
            return types.SimpleNamespace(
                roi_drop=-999.0, error_threshold=999.0, test_failure_threshold=1.0
            )

        def log_evolution_cycle(self, *a, **k) -> None:
            pass

        def check_degradation(self, *_a):
            return True

        def reload_thresholds(self, _bot: str):
            return self.get_thresholds(_bot)

    engine = DummyEngine()

    def record_patch_outcome(session_id, success, contribution=0.0):
        engine.cognition_layer.calls.append((session_id, success, contribution))

    engine.cognition_layer.record_patch_outcome = record_patch_outcome
    pipeline = DummyPipeline()
    data_bot = PredictingDataBot()
    data_bot.check_degradation = lambda *a, **k: True
    data_bot.record_validation = lambda *a, **k: None
    data_bot.log_evolution_cycle = lambda *a, **k: None

    class DummyQuickFix:
        def __init__(self, *a, context_builder=None, **k):
            self.context_builder = context_builder

    monkeypatch.setattr(scm, "QuickFixEngine", DummyQuickFix)
    monkeypatch.setattr(
        scm,
        "generate_patch",
        lambda module_path, *a, **k: (1, []),
    )

    mgr = scm.SelfCodingManager(
        engine,
        pipeline,
        bot_name="bot",
        data_bot=data_bot,
        bot_registry=DummyRegistry(),
    )
    file_path = tmp_path / "sample.py"  # path-ignore
    file_path.write_text("def x():\n    pass\n")
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    tmpdir_path = tmp_path / "clone"

    class DummyTempDir:
        def __enter__(self):
            tmpdir_path.mkdir()
            return str(tmpdir_path)

        def __exit__(self, exc_type, exc, tb):
            shutil.rmtree(tmpdir_path)

    monkeypatch.setattr(tempfile, "TemporaryDirectory", lambda: DummyTempDir())

    def fake_run(cmd, *a, **kw):
        if cmd[:2] == ["git", "clone"]:
            dst = Path(cmd[3])
            dst.mkdir(exist_ok=True)
            shutil.copy2(file_path, dst / file_path.name)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(scm.subprocess, "run", fake_run)
    monkeypatch.setattr(
        scm,
        "run_tests",
        lambda repo, path, **kw: types.SimpleNamespace(
            success=True, failure=None, stdout="", stderr="", duration=0.0
        ),
    )
    monkeypatch.setattr(
        scm.MutationLogger,
        "record_mutation_outcome",
        lambda *a, **k: None,
        raising=False,
    )
    mgr.run_patch(
        file_path, "add", context_meta={"retrieval_session_id": "sid"}
    )
    assert engine.cognition_layer.calls == [("sid", True, pytest.approx(0.0))]


def test_run_patch_quick_fix_success(monkeypatch, tmp_path):
    builder = types.SimpleNamespace(refresh_db_weights=lambda: None)

    class DummyEngine:
        def __init__(self):
            self.cognition_layer = types.SimpleNamespace(context_builder=builder)

    class DummyPipeline:
        def run(self, model: str, energy: int = 1) -> mapl.AutomationResult:
            return mapl.AutomationResult(package=None, roi=None)

    class DummyDataBot:
        def roi(self, _bot: str) -> float:
            return 1.0

        def average_errors(self, _bot: str) -> float:
            return 0.0

        def average_test_failures(self, _bot: str) -> float:
            return 0.0

        def get_thresholds(self, _bot: str):
            return types.SimpleNamespace(
                roi_drop=-999.0, error_threshold=999.0, test_failure_threshold=1.0
            )

        def log_evolution_cycle(self, *a, **k) -> None:
            pass

        def check_degradation(self, *_a):
            return True

        def reload_thresholds(self, _bot: str):
            return self.get_thresholds(_bot)
        def check_degradation(self, *_a):
            return True

        def reload_thresholds(self, _bot: str):
            return self.get_thresholds(_bot)

    class DummyQuickFix:
        def __init__(self, *a, context_builder=None, **k):
            self.context_builder = context_builder

    monkeypatch.setattr(scm, "QuickFixEngine", DummyQuickFix)

    engine = DummyEngine()
    pipeline = DummyPipeline()
    mgr = scm.SelfCodingManager(
        engine,
        pipeline,
        bot_name="bot",
        data_bot=DummyDataBot(),
        bot_registry=DummyRegistry(),
    )

    file_path = tmp_path / "sample.py"  # path-ignore
    file_path.write_text("def x():\n    pass\n")
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    tmpdir_path = tmp_path / "clone"

    class DummyTempDir:
        def __enter__(self):
            tmpdir_path.mkdir()
            return str(tmpdir_path)

        def __exit__(self, exc_type, exc, tb):
            shutil.rmtree(tmpdir_path)

    monkeypatch.setattr(tempfile, "TemporaryDirectory", lambda: DummyTempDir())

    def fake_run(cmd, *a, **kw):
        if cmd[:2] == ["git", "clone"]:
            dst = Path(cmd[3])
            dst.mkdir(exist_ok=True)
            shutil.copy2(file_path, dst / file_path.name)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(scm.subprocess, "run", fake_run)
    monkeypatch.setattr(
        scm.subprocess, "check_output", lambda *a, **k: b"abc123\n"
    )
    monkeypatch.setattr(
        scm,
        "run_tests",
        lambda repo, path, **kw: types.SimpleNamespace(
            success=True, failure=None, stdout="", stderr="", duration=0.0
        ),
    )

    def fake_generate(module_path, *a, **k):
        with open(module_path, "a", encoding="utf-8") as fh:
            fh.write("# patched\n")
        return 123, []

    monkeypatch.setattr(scm, "generate_patch", fake_generate)

    events: list[tuple[str, dict]] = []

    class Bus:
        def publish(self, name: str, payload: dict) -> None:
            events.append((name, payload))

    mgr.event_bus = Bus()

    records: list[tuple] = []
    monkeypatch.setattr(
        scm,
        "record_patch_metadata",
        lambda *a, **k: records.append((a, k)),
    )
    monkeypatch.setattr(
        scm.MutationLogger,
        "record_mutation_outcome",
        lambda *a, **k: None,
        raising=False,
    )

    mgr.run_patch(file_path, "add")
    assert any(
        name == "self_coding:patch_applied"
        and payload.get("commit") == "abc123"
        and payload.get("patch_id") == 123
        for name, payload in events
    )
    assert records[0][0][0] == 123


def test_post_patch_validation_and_hot_swap(monkeypatch, tmp_path):
    builder = types.SimpleNamespace(refresh_db_weights=lambda: None)

    class DummyEngine:
        def __init__(self):
            self.cognition_layer = types.SimpleNamespace(context_builder=builder)

    class DummyPipeline:
        def run(self, model: str, energy: int = 1) -> mapl.AutomationResult:
            return mapl.AutomationResult(package=None, roi=None)

    class DummyDataBot:
        def roi(self, _bot: str) -> float:
            return 1.0

        def average_errors(self, _bot: str) -> float:
            return 0.0

        def average_test_failures(self, _bot: str) -> float:
            return 0.0

        def get_thresholds(self, _bot: str):
            return types.SimpleNamespace(
                roi_drop=-999.0, error_threshold=999.0, test_failure_threshold=1.0
            )

        def log_evolution_cycle(self, *a, **k) -> None:
            pass

        def check_degradation(self, *_a):
            return True

        def reload_thresholds(self, _bot: str):
            return self.get_thresholds(_bot)

        def collect(self, *a, **k):
            pass

    class DummyQuickFix:
        def __init__(self, *a, context_builder=None, **k):
            self.context_builder = context_builder
            self.calls: list[str] = []

        def validate_patch(self, module_name, desc, repo_root=None):
            self.calls.append("validate")
            return True, []

        def apply_validated_patch(self, module_name, desc, ctx_meta=None):
            self.calls.append("apply")
            with open(module_name, "a", encoding="utf-8") as fh:
                fh.write("# patched\n")
            return True, 123, []

    class HotSwapRegistry(DummyRegistry):
        def __init__(self) -> None:
            super().__init__()
            self.hot_swapped: tuple[str, str] | None = None

        def hot_swap(self, name: str, module_path: str) -> None:
            self.hot_swapped = (name, module_path)

        def health_check_bot(self, *a, **k) -> None:
            pass

    engine = DummyEngine()
    pipeline = DummyPipeline()
    registry = HotSwapRegistry()
    qf = DummyQuickFix(context_builder=builder)
    mgr = scm.SelfCodingManager(
        engine,
        pipeline,
        bot_name="bot",
        data_bot=DummyDataBot(),
        bot_registry=registry,
        quick_fix=qf,
    )

    file_path = tmp_path / "sample.py"  # path-ignore
    file_path.write_text("def x():\n    pass\n")
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    tmpdir_path = tmp_path / "clone"

    class DummyTempDir:
        def __enter__(self):
            tmpdir_path.mkdir()
            return str(tmpdir_path)

        def __exit__(self, exc_type, exc, tb):
            shutil.rmtree(tmpdir_path)

    monkeypatch.setattr(tempfile, "TemporaryDirectory", lambda: DummyTempDir())

    def fake_run(cmd, *a, **kw):
        if cmd[:2] == ["git", "clone"]:
            dst = Path(cmd[3])
            dst.mkdir(exist_ok=True)
            shutil.copy2(file_path, dst / file_path.name)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(scm.subprocess, "run", fake_run)
    monkeypatch.setattr(
        scm.subprocess, "check_output", lambda *a, **k: b"abc123\n"
    )
    monkeypatch.setattr(
        scm,
        "run_tests",
        lambda repo, path, **kw: types.SimpleNamespace(
            success=True, failure=None, stdout="", stderr="", duration=0.0
        ),
    )
    monkeypatch.setattr(
        scm.MutationLogger,
        "record_mutation_outcome",
        lambda *a, **k: None,
        raising=False,
    )
    monkeypatch.setattr(scm, "record_patch_metadata", lambda *a, **k: None)

    events: list[tuple[str, dict]] = []

    class Bus:
        def publish(self, name: str, payload: dict) -> None:
            events.append((name, payload))

    mgr.event_bus = Bus()

    mgr.run_patch(file_path, "add")
    assert qf.calls.count("validate") == 2
    assert registry.hot_swapped == ("bot", str(file_path))
    assert any(
        name == "self_coding:patch_applied"
        and payload.get("commit") == "abc123"
        and payload.get("roi_delta") == 0.0
        for name, payload in events
    )


def test_run_patch_quick_fix_failure(monkeypatch, tmp_path):
    builder = types.SimpleNamespace(refresh_db_weights=lambda: None)

    class DummyEngine:
        def __init__(self):
            self.cognition_layer = types.SimpleNamespace(context_builder=builder)

    class DummyPipeline:
        def run(self, model: str, energy: int = 1) -> mapl.AutomationResult:
            return mapl.AutomationResult(package=None, roi=None)

    class DummyDataBot:
        def roi(self, _bot: str) -> float:
            return 1.0

        def average_errors(self, _bot: str) -> float:
            return 0.0

        def average_test_failures(self, _bot: str) -> float:
            return 0.0

        def get_thresholds(self, _bot: str):
            return types.SimpleNamespace(
                roi_drop=-999.0, error_threshold=999.0, test_failure_threshold=1.0
            )

        def log_evolution_cycle(self, *a, **k) -> None:
            pass

        def check_degradation(self, *_a):
            return True

        def reload_thresholds(self, _bot: str):
            return self.get_thresholds(_bot)

    class DummyQuickFix:
        def __init__(self, *a, context_builder=None, **k):
            self.context_builder = context_builder

    monkeypatch.setattr(scm, "QuickFixEngine", DummyQuickFix)

    engine = DummyEngine()
    pipeline = DummyPipeline()
    mgr = scm.SelfCodingManager(
        engine,
        pipeline,
        bot_name="bot",
        data_bot=DummyDataBot(),
        bot_registry=DummyRegistry(),
    )

    file_path = tmp_path / "sample.py"  # path-ignore
    file_path.write_text("def x():\n    pass\n")
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    tmpdir_path = tmp_path / "clone"

    class DummyTempDir:
        def __enter__(self):
            tmpdir_path.mkdir()
            return str(tmpdir_path)

        def __exit__(self, exc_type, exc, tb):
            shutil.rmtree(tmpdir_path)

    monkeypatch.setattr(tempfile, "TemporaryDirectory", lambda: DummyTempDir())

    def fake_run(cmd, *a, **kw):
        if cmd[:2] == ["git", "clone"]:
            dst = Path(cmd[3])
            dst.mkdir(exist_ok=True)
            shutil.copy2(file_path, dst / file_path.name)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(scm.subprocess, "run", fake_run)
    monkeypatch.setattr(
        scm,
        "run_tests",
        lambda repo, path, **kw: types.SimpleNamespace(
            success=True, failure=None, stdout="", stderr="", duration=0.0
        ),
    )

    def bad_generate(module_path, *a, **k):
        with open(module_path, "a", encoding="utf-8") as fh:
            fh.write("# patched\n")
        return None, ["risky"]

    monkeypatch.setattr(scm, "generate_patch", bad_generate)

    events: list[tuple[str, dict]] = []

    class Bus:
        def publish(self, name: str, payload: dict) -> None:
            events.append((name, payload))

    mgr.event_bus = Bus()

    rollbacks: list[str] = []

    class RB:
        def rollback(self, pid: str, requesting_bot: str | None = None) -> None:
            rollbacks.append(pid)

    monkeypatch.setattr(scm, "RollbackManager", lambda: RB())
    monkeypatch.setattr(
        scm.MutationLogger,
        "record_mutation_outcome",
        lambda *a, **k: None,
        raising=False,
    )

    with pytest.raises(RuntimeError):
        mgr.run_patch(file_path, "add")
    assert any(name == "bot:patch_failed" for name, _ in events)
    assert rollbacks


def test_run_patch_requires_quick_fix_engine(monkeypatch, tmp_path):
    builder = types.SimpleNamespace(refresh_db_weights=lambda: None)

    class DummyEngine:
        def __init__(self):
            self.cognition_layer = types.SimpleNamespace(context_builder=builder)

    class DummyPipeline:
        def run(self, model: str, energy: int = 1) -> mapl.AutomationResult:
            return mapl.AutomationResult(package=None, roi=None)

    class DummyDataBot:
        def roi(self, _bot: str) -> float:
            return 1.0

        def average_errors(self, _bot: str) -> float:
            return 0.0

        def average_test_failures(self, _bot: str) -> float:
            return 0.0

        def get_thresholds(self, _bot: str):
            return types.SimpleNamespace(
                roi_drop=-999.0, error_threshold=999.0, test_failure_threshold=1.0
            )

        def log_evolution_cycle(self, *a, **k) -> None:
            pass

        def check_degradation(self, *_a):
            return True

        def reload_thresholds(self, _bot: str):
            return self.get_thresholds(_bot)

    engine = DummyEngine()
    pipeline = DummyPipeline()
    quick_fix = types.SimpleNamespace(
        context_builder=builder, apply_validated_patch=lambda *a, **k: (True, 1, [])
    )
    mgr = scm.SelfCodingManager(
        engine,
        pipeline,
        bot_name="bot",
        data_bot=DummyDataBot(),
        bot_registry=DummyRegistry(),
        quick_fix=quick_fix,
        evolution_orchestrator=types.SimpleNamespace(
            register_bot=lambda *a, **k: None, provenance_token="token"
        ),
    )

    monkeypatch.setattr(scm, "QuickFixEngine", None)
    mgr.quick_fix = None

    file_path = tmp_path / "sample.py"  # path-ignore
    file_path.write_text("def x():\n    pass\n")
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    tmpdir_path = tmp_path / "clone"

    class DummyTempDir:
        def __enter__(self):
            tmpdir_path.mkdir()
            return str(tmpdir_path)

        def __exit__(self, exc_type, exc, tb):
            shutil.rmtree(tmpdir_path)

    monkeypatch.setattr(tempfile, "TemporaryDirectory", lambda: DummyTempDir())
    monkeypatch.setattr(
        scm.subprocess, "run", lambda cmd, *a, **kw: subprocess.CompletedProcess(cmd, 0)
    )

    with pytest.raises(ImportError, match="QuickFixEngine is required"):
        mgr.run_patch(file_path, "add", provenance_token="token")


def test_registry_update_and_hot_swap(monkeypatch, tmp_path):
    class DummyEngine:
        def __init__(self) -> None:
            self.cognition_layer = types.SimpleNamespace(
                context_builder=types.SimpleNamespace()
            )

        def apply_patch(self, path: Path, desc: str, **_: object):
            with open(path, "a", encoding="utf-8") as fh:
                fh.write("# patched\n")
            return 1, False, 0.0

    class DummyPipeline:
        def run(self, model: str, energy: int = 1) -> mapl.AutomationResult:
            return mapl.AutomationResult(package=None, roi=None)

    class DummyDataBot:
        def roi(self, _bot: str) -> float:
            return 1.0

        def average_errors(self, _bot: str) -> float:
            return 0.0

        def average_test_failures(self, _bot: str) -> float:
            return 0.0

        def get_thresholds(self, _bot: str):
            return types.SimpleNamespace(
                roi_drop=-999.0, error_threshold=999.0, test_failure_threshold=1.0
            )

        def check_degradation(self, *_):
            return True

        def log_evolution_cycle(self, *a, **k) -> None:  # pragma: no cover - simple
            pass

    class Graph:
        def __init__(self) -> None:
            self.nodes: dict[str, dict] = {}

        def __contains__(self, name: str) -> bool:
            return name in self.nodes

    class DummyRegistry:
        def __init__(self) -> None:
            self.graph = Graph()
            self.update_args: tuple | None = None
            self.hot_swapped = False
            self.health_checked = False

        def record_heartbeat(self, _name: str) -> None:  # pragma: no cover - simple
            pass

        def register_interaction(self, *_a, **_k) -> None:  # pragma: no cover - simple
            pass

        def record_interaction_metadata(self, *a, **k) -> None:  # pragma: no cover - simple
            pass

        def register_bot(self, name: str) -> None:
            self.graph.nodes.setdefault(name, {})

        def update_bot(self, name: str, module: str, *, patch_id=None, commit=None) -> None:
            self.update_args = (name, module, patch_id, commit)
            self.graph.nodes.setdefault(name, {})["version"] = 1

        def hot_swap_bot(self, name: str) -> None:
            self.hot_swapped = True

        def health_check_bot(self, name: str, prev_state) -> None:  # pragma: no cover - simple
            self.health_checked = True

    engine = DummyEngine()
    pipeline = DummyPipeline()
    registry = DummyRegistry()
    bus = types.SimpleNamespace(events=[], publish=lambda n, p: bus.events.append((n, p)))
    monkeypatch.setattr(scm, "generate_patch", lambda *a, **k: (1, []))
    mgr = scm.SelfCodingManager(
        engine,
        pipeline,
        bot_name="bot",
        data_bot=DummyDataBot(),
        bot_registry=registry,
        quick_fix=types.SimpleNamespace(
            context_builder=None, apply_validated_patch=lambda *a, **k: (True, None, [])
        ),
        event_bus=bus,
    )
    file_path = tmp_path / "sample.py"  # path-ignore
    file_path.write_text("def x():\n    pass\n")
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    tmpdir_path = tmp_path / "clone"

    class DummyTempDir:
        def __enter__(self):
            tmpdir_path.mkdir()
            return str(tmpdir_path)

        def __exit__(self, exc_type, exc, tb):
            shutil.rmtree(tmpdir_path)

    monkeypatch.setattr(tempfile, "TemporaryDirectory", lambda: DummyTempDir())

    def fake_run(cmd, *a, **kw):
        if cmd[:2] == ["git", "clone"]:
            dst = Path(cmd[3])
            dst.mkdir(exist_ok=True)
            shutil.copy2(file_path, dst / file_path.name)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(scm.subprocess, "run", fake_run)
    monkeypatch.setattr(scm.subprocess, "check_output", lambda *a, **k: b"deadbeef")
    monkeypatch.setattr(
        scm,
        "run_tests",
        lambda repo, path, *, backend="venv": types.SimpleNamespace(
            success=True, failure=None, stdout="", stderr="", duration=0.0
        ),
    )
    monkeypatch.setattr(
        scm.MutationLogger,
        "record_mutation_outcome",
        lambda *a, **k: None,
        raising=False,
    )

    mgr.run_patch(file_path, "add")
    assert registry.update_args == (
        "bot",
        scm.path_for_prompt(file_path),
        1,
        "deadbeef",
    )
    assert registry.hot_swapped
    assert registry.health_checked
    event = next((p for n, p in bus.events if n == "bot:updated" and p.get("bot") == "bot"), None)
    assert event is not None and event["patch_id"] == 1 and event["commit"] == "deadbeef"
    assert not any(n == "bot:update_blocked" for n, _ in bus.events)


def test_generate_and_patch_delegates(monkeypatch, tmp_path):
    calls: list[tuple] = []

    class Engine:
        def __init__(self) -> None:
            base_builder = types.SimpleNamespace(refresh_db_weights=lambda: None)
            self.cognition_layer = types.SimpleNamespace(context_builder=base_builder)

        def apply_patch(self, path: Path, desc: str, **_: object):
            return 1, False, 0.0

    engine = Engine()
    pipeline = DummyPipeline()
    mgr = scm.SelfCodingManager(
        engine,
        pipeline,
        bot_name="bot",
        data_bot=DummyDataBot(),
        bot_registry=DummyRegistry(),
        quick_fix=object(),
        evolution_orchestrator=types.SimpleNamespace(
            register_bot=lambda *a, **k: None, provenance_token="token"
        ),
    )
    file_path = tmp_path / "sample.py"
    file_path.write_text("pass\n")

    def fake_run_patch(path, desc, **kw):
        calls.append(("patch", path, desc, kw.get("context_builder")))
        return mapl.AutomationResult(None, prb.ROIResult(0, 0, 0, 0, 0))

    monkeypatch.setattr(mgr, "run_patch", fake_run_patch)
    builder = types.SimpleNamespace(refresh_db_weights=lambda: None)
    monkeypatch.setattr(mgr, "_ensure_quick_fix_engine", lambda *_a, **_k: object())
    mgr.generate_and_patch(file_path, "fix", context_builder=builder)
    assert any(c[0] == "patch" and c[1] == file_path and c[3] is builder for c in calls)


def test_generate_patch_requires_builder(monkeypatch, tmp_path):
    class Engine:
        def __init__(self) -> None:
            base_builder = types.SimpleNamespace(refresh_db_weights=lambda: None)
            self.cognition_layer = types.SimpleNamespace(context_builder=base_builder)

    engine = Engine()
    pipeline = DummyPipeline()
    mgr = scm.SelfCodingManager(
        engine,
        pipeline,
        bot_name="bot",
        data_bot=DummyDataBot(),
        bot_registry=DummyRegistry(),
        quick_fix=object(),
        evolution_orchestrator=types.SimpleNamespace(
            register_bot=lambda *a, **k: None, provenance_token="token"
        ),
    )
    with pytest.raises(TypeError):
        mgr.generate_patch("module.py", provenance_token="token")


def test_generate_and_patch_requires_builder(tmp_path):
    mgr = object.__new__(scm.SelfCodingManager)
    mgr.validate_provenance = lambda _token: None
    file_path = tmp_path / "sample.py"
    file_path.write_text("pass\n")
    with pytest.raises(TypeError):
        scm.SelfCodingManager.generate_and_patch(
            mgr, file_path, "fix", provenance_token="tok"
        )


def test_generate_and_patch_refreshes_builder(monkeypatch, tmp_path):
    calls: list[int] = []

    class Engine:
        def __init__(self) -> None:
            base_builder = types.SimpleNamespace(refresh_db_weights=lambda: None)
            self.cognition_layer = types.SimpleNamespace(context_builder=base_builder)

        def apply_patch(self, path: Path, desc: str, **_: object):  # pragma: no cover - stub
            return 1, False, 0.0

    engine = Engine()
    pipeline = DummyPipeline()
    mgr = scm.SelfCodingManager(
        engine,
        pipeline,
        bot_name="bot",
        data_bot=DummyDataBot(),
        bot_registry=DummyRegistry(),
        quick_fix=object(),
    )
    file_path = tmp_path / "sample.py"
    file_path.write_text("pass\n")

    def fake_run_patch(path, desc, **kw):
        return mapl.AutomationResult(None, prb.ROIResult(0, 0, 0, 0, 0))

    monkeypatch.setattr(mgr, "run_patch", fake_run_patch)
    monkeypatch.setattr(mgr, "_ensure_quick_fix_engine", lambda *_a, **_k: object())

    class Builder:
        def refresh_db_weights(self) -> None:
            calls.append(1)

    builder = Builder()

    mgr.generate_and_patch(file_path, "fix", context_builder=builder)
    mgr.generate_and_patch(file_path, "fix", context_builder=builder)

    assert len(calls) == 2


def test_run_patch_skips_on_low_predicted_roi(monkeypatch, tmp_path):
    events: list[tuple[str, dict]] = []

    class Bus:
        def publish(self, topic, payload):
            events.append((topic, payload))

    class DataBot(DummyDataBot):
        def check_degradation(self, *a, **k):
            return True

        def get_thresholds(self, _bot: str):
            return types.SimpleNamespace(
                roi_drop=-0.1, error_threshold=999.0, test_failure_threshold=1.0
            )

        def reload_thresholds(self, bot: str):
            return self.get_thresholds(bot)

        def forecast_roi_drop(self, limit: int = 100) -> float:
            return -0.2

    data_bot = DataBot()

    class Engine:
        def __init__(self) -> None:
            base_builder = types.SimpleNamespace(refresh_db_weights=lambda: None)
            self.cognition_layer = types.SimpleNamespace(context_builder=base_builder)

        def apply_patch(self, path: Path, desc: str, **_: object):
            with open(path, "a", encoding="utf-8") as fh:
                fh.write("# patched\n")
            return 1, False, 0.0

    pipeline = DummyPipeline()

    called = {"applied": False}

    def apply_validated_patch(*a, **k):
        called["applied"] = True
        return True, None, []

    quick_fix = types.SimpleNamespace(
        context_builder=None, apply_validated_patch=apply_validated_patch
    )

    bus = Bus()
    class Registry:
        def __init__(self) -> None:
            self.graph: dict[str, dict] = {}

        def record_heartbeat(self, _name: str) -> None:
            pass

        def register_interaction(self, *a, **k) -> None:  # pragma: no cover - stub
            pass

        def record_interaction_metadata(self, *a, **k) -> None:  # pragma: no cover - stub
            pass

        def register_bot(self, name: str) -> None:
            self.graph.setdefault(name, {})

        def record_validation(self, *a, **k) -> None:  # pragma: no cover - stub
            pass

        def update_bot(self, *a, **k) -> None:  # pragma: no cover - stub
            pass

        def update_bot(self, *a, **k) -> None:  # pragma: no cover - stub
            pass

    registry = Registry()
    mgr = scm.SelfCodingManager(
        Engine(),
        pipeline,
        bot_name="bot",
        data_bot=data_bot,
        bot_registry=registry,
        quick_fix=quick_fix,
        event_bus=bus,
    )

    file_path = tmp_path / "sample.py"
    file_path.write_text("def x():\n    pass\n")
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    tmpdir_path = tmp_path / "clone"

    class DummyTempDir:
        def __enter__(self):
            tmpdir_path.mkdir()
            return str(tmpdir_path)

        def __exit__(self, exc_type, exc, tb):
            shutil.rmtree(tmpdir_path)

    monkeypatch.setattr(tempfile, "TemporaryDirectory", lambda: DummyTempDir())

    def fake_run(cmd, *a, **kw):
        if cmd[:2] == ["git", "clone"]:
            dst = Path(cmd[3])
            dst.mkdir(exist_ok=True)
            shutil.copy2(file_path, dst / file_path.name)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(scm.subprocess, "run", fake_run)

    res = mgr.run_patch(file_path, "add")
    assert not called["applied"]
    assert (
        "bot:patch_skipped",
        {"bot": "bot", "reason": "roi_prediction"},
    ) in events
    assert isinstance(res, mapl.AutomationResult)
    assert res.package is None and res.roi is None


def test_run_patch_accepts_high_predicted_roi(monkeypatch, tmp_path):
    events: list[tuple[str, dict]] = []

    class Bus:
        def publish(self, topic, payload):
            events.append((topic, payload))

    class DataBot(DummyDataBot):
        def check_degradation(self, *a, **k):
            return True

        def get_thresholds(self, _bot: str):
            return types.SimpleNamespace(
                roi_drop=-0.1, error_threshold=999.0, test_failure_threshold=1.0
            )

        def reload_thresholds(self, bot: str):
            return self.get_thresholds(bot)

        def forecast_roi_drop(self, limit: int = 100) -> float:
            return 0.2

        def record_validation(self, *a, **k) -> None:  # pragma: no cover - stub
            pass

        def record_test_failure(self, *a, **k) -> None:  # pragma: no cover - stub
            pass

        def log_evolution_cycle(self, *a, **k) -> None:  # pragma: no cover - stub
            pass

    data_bot = DataBot()

    class Engine:
        def __init__(self) -> None:
            base_builder = types.SimpleNamespace(refresh_db_weights=lambda: None)
            self.cognition_layer = types.SimpleNamespace(context_builder=base_builder)

        def apply_patch(self, path: Path, desc: str, **_: object):
            with open(path, "a", encoding="utf-8") as fh:
                fh.write("# patched\n")
            return 1, False, 0.0

    pipeline = DummyPipeline()

    called = {"applied": False}

    def apply_validated_patch(*a, **k):
        called["applied"] = True
        return True, None, []

    quick_fix = types.SimpleNamespace(
        context_builder=None, apply_validated_patch=apply_validated_patch
    )

    bus = Bus()

    class Registry:
        def __init__(self) -> None:
            self.graph: dict[str, dict] = {}

        def record_heartbeat(self, _name: str) -> None:
            pass

        def register_interaction(self, *a, **k) -> None:  # pragma: no cover - stub
            pass

        def record_interaction_metadata(self, *a, **k) -> None:  # pragma: no cover - stub
            pass

        def register_bot(self, name: str) -> None:
            self.graph.setdefault(name, {})

        def record_validation(self, *a, **k) -> None:  # pragma: no cover - stub
            pass

        def update_bot(self, *a, **k) -> None:  # pragma: no cover - stub
            pass

    registry = Registry()
    mgr = scm.SelfCodingManager(
        Engine(),
        pipeline,
        bot_name="bot",
        data_bot=data_bot,
        bot_registry=registry,
        quick_fix=quick_fix,
        event_bus=bus,
    )

    file_path = tmp_path / "sample.py"
    file_path.write_text("def x():\n    pass\n")
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    tmpdir_path = tmp_path / "clone"

    class DummyTempDir:
        def __enter__(self):
            tmpdir_path.mkdir()
            return str(tmpdir_path)

        def __exit__(self, exc_type, exc, tb):
            shutil.rmtree(tmpdir_path)

    monkeypatch.setattr(tempfile, "TemporaryDirectory", lambda: DummyTempDir())

    def fake_run(cmd, *a, **kw):
        if cmd[:2] == ["git", "clone"]:
            dst = Path(cmd[3])
            dst.mkdir(exist_ok=True)
            shutil.copy2(file_path, dst / file_path.name)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(scm.subprocess, "run", fake_run)
    monkeypatch.setattr(
        scm,
        "run_tests",
        lambda repo, path, **kw: types.SimpleNamespace(
            success=True, failure=None, stdout="", stderr="", duration=0.0
        ),
    )
    monkeypatch.setattr(
        scm.subprocess, "check_output", lambda *a, **k: b"deadbeef"
    )
    monkeypatch.setattr(
        scm.MutationLogger,
        "record_mutation_outcome",
        lambda *a, **k: None,
        raising=False,
    )

    res = mgr.run_patch(file_path, "add")
    assert called["applied"]
    assert (
        "bot:patch_skipped",
        {"bot": "bot", "reason": "roi_prediction"},
    ) not in events
    assert isinstance(res, mapl.AutomationResult)

def test_should_refactor_on_test_failures_only(monkeypatch):
    class DummyEngine:
        patch_suggestion_db = None

    class DummyPipeline:
        pass

    class DummyDataBot:
        def __init__(self) -> None:
            self.failures = 0

        def roi(self, _bot: str) -> float:
            return 1.0

        def average_errors(self, _bot: str) -> float:
            return 0.0

        def average_test_failures(self, _bot: str) -> float:
            return self.failures

        def get_thresholds(self, _bot: str):
            return types.SimpleNamespace(
                roi_drop=-999.0, error_threshold=999.0, test_failure_threshold=1.0
            )

    data_bot = PredictingDataBot()
    mgr = scm.SelfCodingManager(
        DummyEngine(),
        DummyPipeline(),
        bot_name="bot",
        data_bot=data_bot,
        bot_registry=DummyRegistry(),
        quick_fix=object(),
    )
    mgr._last_errors = data_bot.average_errors("bot")
    assert not mgr.should_refactor()
    data_bot.failures = 5
    assert mgr.should_refactor()


class PredictingDataBot:
    def __init__(self) -> None:
        self.thresholds = types.SimpleNamespace(
            roi_drop=0.0, error_threshold=0.0, test_failure_threshold=0.0
        )
        self.current_roi = 1.0
        self.current_err = 3.0
        self.current_fail = 2.0
        self.anomaly_sensitivity = 1.0
        self.confidence = 0.1

    def roi(self, _bot: str) -> float:
        return self.current_roi

    def average_errors(self, _bot: str) -> float:
        return self.current_err

    def average_test_failures(self, _bot: str) -> float:
        return self.current_fail

    def get_thresholds(self, _bot: str):
        return self.thresholds

    def reload_thresholds(self, _bot: str):  # pragma: no cover - simple stub
        return self.thresholds

    def update_thresholds(
        self,
        _bot: str,
        *,
        roi_drop: float | None = None,
        error_threshold: float | None = None,
        test_failure_threshold: float | None = None,
        forecast: dict | None = None,
    ) -> None:
        self.thresholds = types.SimpleNamespace(
            roi_drop=roi_drop,
            error_threshold=error_threshold,
            test_failure_threshold=test_failure_threshold,
        )
        self.forecast = forecast

    def check_degradation(
        self, _bot: str, _roi: float, _errors: float, _failures: float
    ) -> bool:
        t = self.thresholds
        conf = self.confidence
        return (t.roi_drop or 0.0) < -conf or (t.error_threshold or 0.0) > conf or (
            t.test_failure_threshold or 0.0
        ) > conf


def test_should_refactor_with_negative_prediction(monkeypatch):
    class DummyEngine:
        def __init__(self) -> None:
            builder = types.SimpleNamespace(
                refresh_db_weights=lambda: None, session_id=""
            )
            self.cognition_layer = types.SimpleNamespace(context_builder=builder)
            self.patch_suggestion_db = None

    class DummyPipeline:
        pass

    data_bot = PredictingDataBot()
    mgr = scm.SelfCodingManager(
        DummyEngine(),
        DummyPipeline(),
        bot_name="bot",
        data_bot=data_bot,
        bot_registry=DummyRegistry(),
        quick_fix=object(),
    )
    mgr.baseline_tracker.update(roi=3.0, errors=1.0, tests_failed=0.0)
    mgr.baseline_tracker.update(roi=2.0, errors=2.0, tests_failed=1.0)
    assert mgr.should_refactor()
    assert data_bot.thresholds.roi_drop < 0.0
    assert data_bot.thresholds.error_threshold > 0.0
    assert data_bot.thresholds.test_failure_threshold > 0.0


def test_should_refactor_ignores_positive_prediction(monkeypatch):
    class DummyEngine:
        def __init__(self) -> None:
            builder = types.SimpleNamespace(
                refresh_db_weights=lambda: None, session_id=""
            )
            self.cognition_layer = types.SimpleNamespace(context_builder=builder)
            self.patch_suggestion_db = None

    class DummyPipeline:
        pass

    data_bot = PredictingDataBot()
    mgr = scm.SelfCodingManager(
        DummyEngine(),
        DummyPipeline(),
        bot_name="bot",
        data_bot=data_bot,
        bot_registry=DummyRegistry(),
        quick_fix=object(),
    )
    # Positive ROI and stable errors/failures should not trigger refactor
    mgr.baseline_tracker.update(roi=1.0, errors=1.0, tests_failed=0.0)
    mgr.baseline_tracker.update(roi=2.0, errors=0.5, tests_failed=0.0)
    data_bot.current_roi = 3.0
    data_bot.current_err = 0.4
    data_bot.current_fail = 0.0
    assert not mgr.should_refactor()


def test_ema_forecast_reduces_false_positive(monkeypatch):
    class DummyDataBot:
        def __init__(self) -> None:
            self.rois = [1.0, 1.0, 1.0, 0.95]
            self.idx = 0
            self.thresholds = types.SimpleNamespace(
                roi_drop=-0.1, error_threshold=1.0, test_failure_threshold=1.0
            )

        def roi(self, _bot: str) -> float:
            v = self.rois[self.idx]
            self.idx += 1
            return v

        def average_errors(self, _bot: str) -> float:
            return 0.0

        def average_test_failures(self, _bot: str) -> float:
            return 0.0

        def get_thresholds(self, _bot: str):
            return self.thresholds

        def update_thresholds(self, _bot: str, *, roi_drop=None, error_threshold=None, test_failure_threshold=None, forecast=None):
            self.thresholds = types.SimpleNamespace(
                roi_drop=roi_drop,
                error_threshold=error_threshold,
                test_failure_threshold=test_failure_threshold,
            )
            self.forecast = forecast

        def check_degradation(self, _bot, roi, _err, _fail):
            delta = roi - 1.0
            return delta <= (self.thresholds.roi_drop or 0.0)

    class LocalEngine:
        def __init__(self) -> None:
            builder = types.SimpleNamespace(session_id="", refresh_db_weights=lambda: None)
            self.cognition_layer = types.SimpleNamespace(context_builder=builder)
            self.patch_suggestion_db = None

    mgr = scm.SelfCodingManager(
        LocalEngine(),
        DummyPipeline(),
        bot_name="bot",
        data_bot=DummyDataBot(),
        bot_registry=DummyRegistry(),
        quick_fix=types.SimpleNamespace(context_builder=None),
    )
    for _ in range(3):
        assert not mgr.should_refactor()
    assert not mgr.should_refactor()
    assert mgr.data_bot.forecast["roi"]

def test_init_requires_helpers():
    engine = DummyEngine()
    pipeline = DummyPipeline()
    with pytest.raises(ValueError):
        scm.SelfCodingManager(
            engine,
            pipeline,
            bot_name="bot",
            data_bot=DummyDataBot(),
        )
    with pytest.raises(ValueError):
        scm.SelfCodingManager(
            engine,
            pipeline,
            bot_name="bot",
            bot_registry=DummyRegistry(),
        )


def test_update_blocked_event_when_provenance_missing(monkeypatch, tmp_path):
    class DummyEngine:
        def __init__(self) -> None:
            self.cognition_layer = types.SimpleNamespace(context_builder=types.SimpleNamespace())
        def apply_patch(self, path: Path, desc: str, **_: object):
            with open(path, "a", encoding="utf-8") as fh:
                fh.write("# patched\n")
            return 1, False, 0.0

    class DummyPipeline:
        def run(self, model: str, energy: int = 1) -> mapl.AutomationResult:
            return mapl.AutomationResult(package=None, roi=None)

    class DummyDataBot:
        def roi(self, _bot: str) -> float: return 1.0
        def average_errors(self, _bot: str) -> float: return 0.0
        def average_test_failures(self, _bot: str) -> float: return 0.0
        def get_thresholds(self, _bot: str):
            return types.SimpleNamespace(roi_drop=-999.0, error_threshold=999.0, test_failure_threshold=1.0)
        def check_degradation(self, *_): return True
        def log_evolution_cycle(self, *a, **k) -> None: pass  # pragma: no cover - simple

    class Graph:
        def __init__(self) -> None:
            self.nodes: dict[str, dict] = {}
        def __contains__(self, name: str) -> bool:
            return name in self.nodes

    class DummyRegistry:
        def __init__(self) -> None:
            self.graph = Graph()
            self.update_args = None
            self.hot_swapped = False
            self.health_checked = False
        def record_heartbeat(self, _name: str) -> None: pass  # pragma: no cover - simple
        def register_interaction(self, *_a, **_k) -> None: pass  # pragma: no cover - simple
        def record_interaction_metadata(self, *a, **k) -> None: pass  # pragma: no cover - simple
        def register_bot(self, name: str) -> None: self.graph.nodes.setdefault(name, {})
        def update_bot(self, *a, **k) -> None: self.update_args = (a, k)
        def hot_swap_bot(self, *a, **k) -> None: self.hot_swapped = True
        def health_check_bot(self, *a, **k) -> None: self.health_checked = True

    engine = DummyEngine()
    pipeline = DummyPipeline()
    registry = DummyRegistry()
    bus = types.SimpleNamespace(events=[], publish=lambda n, p: bus.events.append((n, p)))
    monkeypatch.setattr(scm, "generate_patch", lambda *a, **k: (1, []))
    mgr = scm.SelfCodingManager(
        engine,
        pipeline,
        bot_name="bot",
        data_bot=DummyDataBot(),
        bot_registry=registry,
        quick_fix=types.SimpleNamespace(
            context_builder=None, apply_validated_patch=lambda *a, **k: (True, None, [])
        ),
        event_bus=bus,
    )
    file_path = tmp_path / "sample.py"  # path-ignore
    file_path.write_text("def x():\n    pass\n")
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
    tmpdir_path = tmp_path / "clone"
    class DummyTempDir:
        def __enter__(self):
            tmpdir_path.mkdir()
            return str(tmpdir_path)
        def __exit__(self, exc_type, exc, tb):
            shutil.rmtree(tmpdir_path)
    monkeypatch.setattr(tempfile, "TemporaryDirectory", lambda: DummyTempDir())
    def fake_run(cmd, *a, **kw):
        if cmd[:2] == ["git", "clone"]:
            dst = Path(cmd[3]); dst.mkdir(exist_ok=True); shutil.copy2(file_path, dst / file_path.name)
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(scm.subprocess, "run", fake_run)
    def bad_check_output(*a, **k):
        raise subprocess.CalledProcessError(1, a[0])
    monkeypatch.setattr(scm.subprocess, "check_output", bad_check_output)
    monkeypatch.setattr(
        scm,
        "run_tests",
        lambda repo, path, *, backend="venv": types.SimpleNamespace(success=True, failure=None, stdout="", stderr="", duration=0.0),
    )
    monkeypatch.setattr(
        scm.MutationLogger,
        "record_mutation_outcome",
        lambda *a, **k: None,
        raising=False,
    )
    mgr.run_patch(file_path, "add")
    assert registry.update_args is None
    assert not registry.hot_swapped
    assert not registry.health_checked
    event = next((p for n, p in bus.events if n == "bot:update_blocked"), None)
    assert event is not None and event["bot"] == "bot"

def test_run_post_patch_cycle_attempts_repair(monkeypatch, tmp_path):
    monkeypatch.setenv("SELF_TEST_REPAIR_RETRIES", "1")

    class DummySelfTestService:
        call_kwargs = []
        results_sequence = []
        call_index = 0

        def __init__(self, **kwargs):
            DummySelfTestService.call_kwargs.append(dict(kwargs))

        def run_once(self):
            idx = DummySelfTestService.call_index
            DummySelfTestService.call_index += 1
            return DummySelfTestService.results_sequence[idx]

    stub_module = types.ModuleType("menace.self_test_service")
    stub_module.SelfTestService = DummySelfTestService
    monkeypatch.setitem(sys.modules, "menace.self_test_service", stub_module)
    monkeypatch.setitem(sys.modules, "self_test_service", stub_module)

    class DummyQuickFix:
        def __init__(self):
            self.context_builder = None
            self.apply_calls = []
            self.validate_calls = []

        def validate_patch(self, module_name, description, **_kw):
            self.validate_calls.append({"module": module_name, "description": description})
            return True, []

        def apply_validated_patch(self, module_name, description, ctx_meta=None, provenance_token=None):
            self.apply_calls.append(
                {
                    "module": module_name,
                    "description": description,
                    "ctx_meta": dict(ctx_meta or {}),
                    "token": provenance_token,
                }
            )
            return True, 321, []

    quick_fix = DummyQuickFix()

    def make_manager():
        builder = types.SimpleNamespace(refresh_db_weights=lambda: None, session_id=None)

        class Engine:
            def __init__(self):
                self.cognition_layer = types.SimpleNamespace(context_builder=builder)
                self.patch_db = None

        class Pipeline:
            workflow_test_args = ["tests/test_mod.py"]

        class DataBot:
            def __init__(self):
                self.collect_calls = []
                self.validation_calls = []

            def collect(self, bot, **metrics):
                self.collect_calls.append((bot, dict(metrics)))

            def record_validation(self, bot, module, passed, flags=None):
                self.validation_calls.append((bot, module, passed, list(flags or [])))

            def roi(self, _bot):
                return 0.0

            def average_errors(self, _bot):
                return 0.0

        class Registry:
            def register_bot(self, *args, **kwargs):
                return None

        data_bot = DataBot()
        engine = Engine()
        pipeline = Pipeline()
        registry = Registry()
        monkeypatch.setattr(
            scm,
            "SandboxSettings",
            lambda: types.SimpleNamespace(
                baseline_window=5,
                self_test_repair_retries=None,
                post_patch_repair_attempts=None,
            ),
            raising=False,
        )
        monkeypatch.setattr(
            scm,
            "create_context_builder",
            lambda: types.SimpleNamespace(refresh_db_weights=lambda: None, session_id=None),
        )
        monkeypatch.setattr(scm, "ensure_fresh_weights", lambda builder: None)
        orchestrator = types.SimpleNamespace(provenance_token="token")
        manager = scm.SelfCodingManager(
            engine,
            pipeline,
            bot_name="bot",
            data_bot=data_bot,
            bot_registry=registry,
            quick_fix=quick_fix,
            evolution_orchestrator=orchestrator,
        )
        return manager, data_bot

    manager, data_bot = make_manager()
    module_path = tmp_path / "module.py"
    module_path.write_text("value = 1\n")
    monkeypatch.chdir(tmp_path)

    def fake_run(cmd, *a, **kw):
        if cmd[:2] == ["git", "clone"]:
            dst = Path(cmd[3])
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(module_path, dst / module_path.name)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(scm.subprocess, "run", fake_run)

    DummySelfTestService.results_sequence = [
        (
            {
                "passed": 0,
                "failed": 1,
                "coverage": 0.0,
                "runtime": 1.0,
                "stdout": "FAILED tests/test_mod.py::test_feature - AssertionError\n",
                "module_metrics": {"tests/test_mod.py": {"categories": ["failed"]}},
            },
            [],
        ),
        (
            {
                "passed": 1,
                "failed": 0,
                "coverage": 1.0,
                "runtime": 0.5,
            },
            [],
        ),
    ]
    DummySelfTestService.call_kwargs = []
    DummySelfTestService.call_index = 0

    summary = manager.run_post_patch_cycle(
        module_path,
        "initial change",
        provenance_token="token",
    )

    assert summary["self_tests"]["failed"] == 0
    assert summary["self_tests"]["attempts"] == 2
    assert "diagnostics" not in summary["self_tests"]
    attempts = summary["self_tests"]["repair_attempts"]
    assert len(attempts) == 1
    assert attempts[0]["node_ids"] == ["tests/test_mod.py::test_feature"]
    assert DummySelfTestService.call_kwargs[0]["pytest_args"] == "tests/test_mod.py"
    assert DummySelfTestService.call_kwargs[1]["pytest_args"] == "tests/test_mod.py::test_feature"
    assert len(quick_fix.apply_calls) == 2
    repair_call = quick_fix.apply_calls[1]
    assert "Repair attempt 1" in repair_call["description"]
    assert "test_feature" in repair_call["description"]
    assert repair_call["ctx_meta"]["repair_attempt"] == 1
    assert repair_call["ctx_meta"]["repair_node_ids"] == ["tests/test_mod.py::test_feature"]
    assert data_bot.validation_calls[-1][2] is True
    assert summary["quick_fix"]["repair_attempts"] == attempts
    assert manager._last_validation_summary == summary


def test_run_post_patch_cycle_escalates_after_budget(monkeypatch, tmp_path):
    monkeypatch.setenv("SELF_TEST_REPAIR_RETRIES", "0")

    class DummySelfTestService:
        call_kwargs = []
        results_sequence = []
        call_index = 0

        def __init__(self, **kwargs):
            DummySelfTestService.call_kwargs.append(dict(kwargs))

        def run_once(self):
            idx = DummySelfTestService.call_index
            DummySelfTestService.call_index += 1
            return DummySelfTestService.results_sequence[idx]

    stub_module = types.ModuleType("menace.self_test_service")
    stub_module.SelfTestService = DummySelfTestService
    monkeypatch.setitem(sys.modules, "menace.self_test_service", stub_module)
    monkeypatch.setitem(sys.modules, "self_test_service", stub_module)

    class DummyQuickFix:
        def __init__(self):
            self.context_builder = None
            self.apply_calls = []

        def validate_patch(self, *_a, **_k):
            return True, []

        def apply_validated_patch(self, module_name, description, ctx_meta=None, provenance_token=None):
            self.apply_calls.append(
                {
                    "module": module_name,
                    "description": description,
                    "ctx_meta": dict(ctx_meta or {}),
                    "token": provenance_token,
                }
            )
            return True, 654, []

    quick_fix = DummyQuickFix()

    def make_manager():
        builder = types.SimpleNamespace(refresh_db_weights=lambda: None, session_id=None)

        class Engine:
            def __init__(self):
                self.cognition_layer = types.SimpleNamespace(context_builder=builder)
                self.patch_db = None

        class Pipeline:
            workflow_test_args = ["tests/test_mod.py"]

        class DataBot:
            def __init__(self):
                self.collect_calls = []
                self.validation_calls = []

            def collect(self, bot, **metrics):
                self.collect_calls.append((bot, dict(metrics)))

            def record_validation(self, bot, module, passed, flags=None):
                self.validation_calls.append((bot, module, passed, list(flags or [])))

            def roi(self, _bot):
                return 0.0

            def average_errors(self, _bot):
                return 0.0

        class Registry:
            def register_bot(self, *args, **kwargs):
                return None

        data_bot = DataBot()
        engine = Engine()
        pipeline = Pipeline()
        registry = Registry()
        monkeypatch.setattr(
            scm,
            "SandboxSettings",
            lambda: types.SimpleNamespace(
                baseline_window=5,
                self_test_repair_retries=None,
                post_patch_repair_attempts=None,
            ),
            raising=False,
        )
        monkeypatch.setattr(
            scm,
            "create_context_builder",
            lambda: types.SimpleNamespace(refresh_db_weights=lambda: None, session_id=None),
        )
        monkeypatch.setattr(scm, "ensure_fresh_weights", lambda builder: None)
        orchestrator = types.SimpleNamespace(provenance_token="token")
        manager = scm.SelfCodingManager(
            engine,
            pipeline,
            bot_name="bot",
            data_bot=data_bot,
            bot_registry=registry,
            quick_fix=quick_fix,
            evolution_orchestrator=orchestrator,
        )
        return manager, data_bot

    manager, data_bot = make_manager()
    module_path = tmp_path / "module.py"
    module_path.write_text("value = 1\n")
    monkeypatch.chdir(tmp_path)

    def fake_run(cmd, *a, **kw):
        if cmd[:2] == ["git", "clone"]:
            dst = Path(cmd[3])
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(module_path, dst / module_path.name)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(scm.subprocess, "run", fake_run)

    DummySelfTestService.results_sequence = [
        (
            {
                "passed": 0,
                "failed": 1,
                "coverage": 0.0,
                "runtime": 1.0,
                "stdout": "FAILED tests/test_mod.py::test_feature - AssertionError\n",
                "module_metrics": {"tests/test_mod.py": {"categories": ["failed"]}},
            },
            [],
        )
    ]
    DummySelfTestService.call_kwargs = []
    DummySelfTestService.call_index = 0

    with pytest.raises(RuntimeError) as excinfo:
        manager.run_post_patch_cycle(
            module_path,
            "initial change",
            provenance_token="token",
        )

    assert "self tests failed (1)" in str(excinfo.value)
    summary = manager._last_validation_summary
    assert summary["self_tests"]["failed"] == 1
    diagnostics = summary["self_tests"].get("diagnostics")
    assert diagnostics
    assert diagnostics["node_ids"] == ["tests/test_mod.py::test_feature"]
    assert summary["self_tests"]["repair_attempts"] == []
    assert len(quick_fix.apply_calls) == 1
    assert DummySelfTestService.call_kwargs[0]["pytest_args"] == "tests/test_mod.py"
    assert data_bot.validation_calls == []



def test_resolve_manager_timeout_seconds_botplanningbot_override(monkeypatch):
    monkeypatch.setattr(scm, "_MANAGER_CONSTRUCTION_TIMEOUT_SECONDS", 45.0)
    monkeypatch.setenv(
        "SELF_CODING_MANAGER_CONSTRUCTION_TIMEOUT_SECONDS_BOTPLANNINGBOT", "123"
    )

    assert scm._resolve_manager_timeout_seconds("BotPlanningBot") == 123.0


def test_resolve_manager_timeout_seconds_per_bot_override_precedence(monkeypatch):
    monkeypatch.setattr(scm, "_MANAGER_CONSTRUCTION_TIMEOUT_SECONDS", 45.0)
    monkeypatch.setattr(
        scm,
        "_BOTPLANNINGBOT_MANAGER_CONSTRUCTION_TIMEOUT_FALLBACK_SECONDS",
        105.0,
    )
    monkeypatch.setenv(
        "SELF_CODING_MANAGER_CONSTRUCTION_TIMEOUT_SECONDS_BOTPLANNINGBOT", "110"
    )
    monkeypatch.setenv(
        "SELF_CODING_MANAGER_TIMEOUT_SECONDS_BOTPLANNINGBOT", "135"
    )

    assert scm._resolve_manager_timeout_seconds("BotPlanningBot") == 110.0


def test_resolve_manager_timeout_seconds_invalid_env_value_falls_back_safely(monkeypatch):
    monkeypatch.setattr(scm, "_MANAGER_CONSTRUCTION_TIMEOUT_SECONDS", 45.0)
    monkeypatch.setattr(
        scm,
        "_BOTPLANNINGBOT_MANAGER_CONSTRUCTION_TIMEOUT_FALLBACK_SECONDS",
        105.0,
    )
    monkeypatch.setenv(
        "SELF_CODING_MANAGER_CONSTRUCTION_TIMEOUT_SECONDS_BOTPLANNINGBOT", "invalid"
    )
    monkeypatch.setenv(
        "SELF_CODING_MANAGER_TIMEOUT_SECONDS_BOTPLANNINGBOT", "118"
    )

    assert scm._resolve_manager_timeout_seconds("BotPlanningBot") == 118.0


def test_resolve_manager_timeout_seconds_botplanningbot_default_fallback(monkeypatch):
    monkeypatch.setattr(scm, "_MANAGER_CONSTRUCTION_TIMEOUT_SECONDS", 45.0)
    monkeypatch.setattr(
        scm,
        "_BOTPLANNINGBOT_MANAGER_CONSTRUCTION_TIMEOUT_FALLBACK_SECONDS",
        105.0,
    )
    monkeypatch.delenv(
        "SELF_CODING_MANAGER_CONSTRUCTION_TIMEOUT_SECONDS_BOTPLANNINGBOT",
        raising=False,
    )
    monkeypatch.delenv(
        "SELF_CODING_MANAGER_TIMEOUT_SECONDS_BOTPLANNINGBOT",
        raising=False,
    )

    assert scm._resolve_manager_timeout_seconds("BotPlanningBot") == 105.0


def test_resolve_manager_timeout_seconds_warns_on_low_heavy_bot_timeout(
    monkeypatch, caplog
):
    monkeypatch.setattr(scm, "_MANAGER_CONSTRUCTION_TIMEOUT_SECONDS", 45.0)
    monkeypatch.setattr(
        scm,
        "_BOTPLANNINGBOT_MANAGER_CONSTRUCTION_TIMEOUT_FALLBACK_SECONDS",
        105.0,
    )
    monkeypatch.setattr(scm, "_HEAVY_MANAGER_TIMEOUT_MIN_SECONDS", 90.0)
    monkeypatch.setenv(
        "SELF_CODING_MANAGER_CONSTRUCTION_TIMEOUT_SECONDS_BOTPLANNINGBOT", "60"
    )

    with caplog.at_level("WARNING"):
        assert scm._resolve_manager_timeout_seconds("BotPlanningBot") == 60.0

    assert "below recommended 90.00s" in caplog.text





def test_internalize_debounce_reuses_existing_manager(monkeypatch):
    class DummyRegistry:
        class Graph:
            def __init__(self):
                self.nodes = {}

        def __init__(self):
            self.graph = self.Graph()

    existing_manager = types.SimpleNamespace(
        quick_fix=object(),
        event_bus=object(),
        logger=logging.getLogger(__name__),
    )
    registry = DummyRegistry()
    registry.graph.nodes["DebouncedBot"] = {"selfcoding_manager": existing_manager}

    monkeypatch.setattr(scm, "_INTERNALIZE_LAST_ATTEMPT_STARTED_AT", {"DebouncedBot": scm.time.monotonic()})
    monkeypatch.setattr(scm, "_INTERNALIZE_DEBOUNCE_SECONDS", 60.0)

    manager = scm.internalize_coding_bot(
        "DebouncedBot",
        object(),
        object(),
        data_bot=types.SimpleNamespace(),
        bot_registry=registry,
        provenance_token="token",
    )

    assert manager is existing_manager
    node = registry.graph.nodes["DebouncedBot"]
    assert node["attempt_result"] == "skipped_debounce"
    assert node["attempt_finished_at"] is not None


def test_internalize_duplicate_invoke_debounce_records_skip_reason(monkeypatch):
    class DummyRegistry:
        class Graph:
            def __init__(self):
                self.nodes = {}

        def __init__(self):
            self.graph = self.Graph()

    existing_manager = types.SimpleNamespace(
        quick_fix=object(),
        event_bus=object(),
        logger=logging.getLogger(__name__),
    )
    registry = DummyRegistry()
    registry.graph.nodes["DupBot"] = {"selfcoding_manager": existing_manager}

    now = scm.time.monotonic()
    monkeypatch.setattr(
        scm,
        "_INTERNALIZE_LAST_MANAGER_CONSTRUCTION_STARTED_AT",
        {"DupBot": now - 0.05},
    )
    monkeypatch.setattr(scm, "_INTERNALIZE_DUPLICATE_INVOKE_DEBOUNCE_SECONDS", 1.0)
    monkeypatch.setattr(scm, "_INTERNALIZE_LAST_ATTEMPT_STARTED_AT", {})
    monkeypatch.setattr(scm, "_INTERNALIZE_DEBOUNCE_SECONDS", 60.0)

    manager = scm.internalize_coding_bot(
        "DupBot",
        object(),
        object(),
        data_bot=types.SimpleNamespace(),
        bot_registry=registry,
        provenance_token="token",
    )

    assert manager is existing_manager
    node = registry.graph.nodes["DupBot"]
    assert node["attempt_result"] == "duplicate_invoke_debounced"
    assert node["attempt_finished_at"] is not None



def test_internalize_active_token_dedup_logs_age_and_reason(monkeypatch, caplog):
    class DummyRegistry:
        class Graph:
            def __init__(self):
                self.nodes = {}

        def __init__(self):
            self.graph = self.Graph()

    existing_manager = types.SimpleNamespace(
        quick_fix=object(),
        event_bus=object(),
        logger=logging.getLogger(__name__),
    )
    registry = DummyRegistry()
    registry.graph.nodes["TokenBot"] = {"selfcoding_manager": existing_manager}

    now = scm.time.monotonic()
    monkeypatch.setattr(
        scm,
        "_INTERNALIZE_ACTIVE_TOKENS",
        {
            "TokenBot": {
                "token": "existing-token",
                "reason": "manager_construction",
                "started_at": now - 4.0,
            }
        },
    )

    with caplog.at_level(logging.INFO):
        manager = scm.internalize_coding_bot(
            "TokenBot",
            object(),
            object(),
            data_bot=types.SimpleNamespace(),
            bot_registry=registry,
            provenance_token="token",
        )

    assert manager is existing_manager
    node = registry.graph.nodes["TokenBot"]
    assert node["attempt_result"] == "active_token_dedup"
    dedup_records = [
        record for record in caplog.records if getattr(record, "event", "") == "internalize_active_token_dedup"
    ]
    assert dedup_records
    record = dedup_records[-1]
    assert getattr(record, "bot", None) == "TokenBot"
    assert getattr(record, "reason", None) == "manager_construction"
    assert getattr(record, "active_attempt_age_seconds", 0.0) >= 0.0


def test_internalize_active_token_cleared_on_early_return(monkeypatch):
    class DummyRegistry:
        class Graph:
            def __init__(self):
                self.nodes = {}

        def __init__(self):
            self.graph = self.Graph()

    existing_manager = types.SimpleNamespace(
        quick_fix=object(),
        event_bus=object(),
        logger=logging.getLogger(__name__),
    )
    registry = DummyRegistry()
    registry.graph.nodes["EarlyReturnBot"] = {"selfcoding_manager": existing_manager}

    monkeypatch.setattr(scm, "_INTERNALIZE_LAST_ATTEMPT_STARTED_AT", {})
    monkeypatch.setattr(scm, "_INTERNALIZE_ACTIVE_TOKENS", {})
    monkeypatch.setattr(scm, "_INTERNALIZE_REUSE_WINDOW_SECONDS", 90.0)

    manager = scm.internalize_coding_bot(
        "EarlyReturnBot",
        object(),
        object(),
        data_bot=types.SimpleNamespace(),
        bot_registry=registry,
        provenance_token="token",
    )

    assert manager is existing_manager
    assert "EarlyReturnBot" not in scm._INTERNALIZE_ACTIVE_TOKENS

def test_internalize_records_attempt_timestamps_and_result(monkeypatch, tmp_path):
    class DummyRegistry:
        class Graph:
            def __init__(self):
                self.nodes = {}

        def __init__(self):
            self.graph = self.Graph()
            self.event_bus = None
            self.modules = {}

        def register_bot(self, _bot_name, **_kwargs):
            return None

    class DummyManager:
        def __init__(self, *args, **kwargs):
            self.quick_fix = object()
            self.event_bus = None
            self.logger = logging.getLogger(__name__)
            self.data_bot = kwargs.get("data_bot")
            self.evolution_orchestrator = None

        def initialize_deferred_components(self, skip_non_critical=False):
            return None

        def run_post_patch_cycle(self, *args, **kwargs):
            return {"self_tests": {"failed": 0}}

    module_path = tmp_path / "bot_module.py"
    module_path.write_text("x = 1\n", encoding="utf-8")
    registry = DummyRegistry()
    registry.graph.nodes["ObserveBot"] = {"module": str(module_path)}

    monkeypatch.setattr(scm, "SelfCodingManager", DummyManager)
    monkeypatch.setattr(scm, "persist_sc_thresholds", lambda *a, **k: None)
    monkeypatch.setattr(scm, "_INTERNALIZE_LAST_ATTEMPT_STARTED_AT", {})

    manager = scm.internalize_coding_bot(
        "ObserveBot",
        object(),
        object(),
        data_bot=types.SimpleNamespace(),
        bot_registry=registry,
        provenance_token="token",
    )

    assert isinstance(manager, DummyManager)
    node = registry.graph.nodes["ObserveBot"]
    assert isinstance(node.get("attempt_started_at"), (int, float))
    assert isinstance(node.get("attempt_finished_at"), (int, float))
    assert node["attempt_finished_at"] >= node["attempt_started_at"]
    assert node.get("attempt_result") == "success"
def test_internalize_manager_timeout_emits_internalization_failure(monkeypatch):
    class DummyEventBus:
        def __init__(self):
            self.published = []

        def publish(self, topic, payload):
            self.published.append((topic, payload))

    class DummyRegistry:
        class Graph:
            def __init__(self):
                self.nodes = {}

        def __init__(self):
            self.graph = self.Graph()
            self.retry_calls = []
            self.event_bus = DummyEventBus()

        def force_internalization_retry(self, bot_name, delay=0.0):
            self.retry_calls.append((bot_name, delay))

    class TimeoutFuture:
        def result(self, timeout=None):
            raise scm.concurrent.futures.TimeoutError()

    class TimeoutExecutor:
        def __init__(self, max_workers=1):
            self.max_workers = max_workers

        def submit(self, func):
            return TimeoutFuture()

        def shutdown(self, wait=False, cancel_futures=False):
            return None

    registry = DummyRegistry()
    degraded = types.SimpleNamespace(degraded=True)
    monkeypatch.setattr(scm, "_MANAGER_CONSTRUCTION_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(scm.concurrent.futures, "ThreadPoolExecutor", TimeoutExecutor)
    monkeypatch.setattr(
        scm,
        "_cooldown_disabled_manager",
        lambda bot_registry, data_bot: degraded,
    )

    manager = scm.internalize_coding_bot(
        "AnyOtherBot",
        object(),
        object(),
        data_bot=types.SimpleNamespace(),
        bot_registry=registry,
        provenance_token="token",
    )

    assert manager is degraded
    failure_events = [
        payload
        for topic, payload in registry.event_bus.published
        if topic == "self_coding:internalization_failure"
    ]
    assert failure_events
    assert failure_events[-1]["reason"] == "manager_construction_timeout"
    assert failure_events[-1]["attempt_index"] == 1
    assert failure_events[-1]["phase_elapsed_seconds"] is not None
    assert "phase_history" in failure_events[-1]


def test_internalize_manager_timeout_botplanningbot_retries_with_bounded_timeout(monkeypatch, caplog):
    class DummyEventBus:
        def __init__(self):
            self.published = []

        def publish(self, topic, payload):
            self.published.append((topic, payload))

    class DummyRegistry:
        class Graph:
            def __init__(self):
                self.nodes = {}

        def __init__(self):
            self.graph = self.Graph()
            self.retry_calls = []
            self.event_bus = DummyEventBus()

        def force_internalization_retry(self, bot_name, delay=0.0):
            self.retry_calls.append((bot_name, delay))

    class TimeoutFuture:
        def __init__(self, record):
            self.record = record

        def result(self, timeout=None):
            self.record.append(timeout)
            raise scm.concurrent.futures.TimeoutError()

    class TimeoutExecutor:
        recorded_timeouts = []

        def __init__(self, max_workers=1):
            self.max_workers = max_workers

        def submit(self, func):
            return TimeoutFuture(self.recorded_timeouts)

        def shutdown(self, wait=False, cancel_futures=False):
            return None

    registry = DummyRegistry()
    degraded = types.SimpleNamespace(degraded=True)
    monkeypatch.setattr(scm, "_MANAGER_CONSTRUCTION_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(
        scm,
        "_resolve_manager_timeout_seconds",
        lambda bot_name: 0.01,
    )
    monkeypatch.setattr(
        scm,
        "_resolve_manager_retry_timeout_seconds",
        lambda bot_name, primary_timeout: 0.01,
    )
    monkeypatch.setattr(scm.concurrent.futures, "ThreadPoolExecutor", TimeoutExecutor)
    monkeypatch.setattr(
        scm,
        "_cooldown_disabled_manager",
        lambda bot_registry, data_bot: degraded,
    )

    manager = scm.internalize_coding_bot(
        "BotPlanningBot",
        object(),
        object(),
        data_bot=types.SimpleNamespace(),
        bot_registry=registry,
        provenance_token="token",
    )

    assert manager is degraded
    failure_events = [
        payload
        for topic, payload in registry.event_bus.published
        if topic == "self_coding:internalization_failure"
    ]
    assert len(failure_events) >= 2
    assert failure_events[0]["retry_timeout_seconds"] == 0.02
    assert failure_events[0]["attempt_index"] == 1
    assert failure_events[-1]["attempt_index"] == 2
    assert failure_events[-1]["fallback_used"] is True
    assert TimeoutExecutor.recorded_timeouts[:2] == [0.01, 0.02]

    retry_failure_logs = [
        record
        for record in caplog.records
        if getattr(record, "event", None)
        == "internalize_manager_construction_timeout_retry_failed"
    ]
    assert retry_failure_logs
    retry_failure = retry_failure_logs[-1]
    assert retry_failure.bot == "BotPlanningBot"
    assert retry_failure.phase
    assert retry_failure.timeout_seconds == 0.01
    assert retry_failure.retry_timeout_seconds == 0.02
    assert retry_failure.fallback_available is True


def test_internalize_manager_timeout_botplanningbot_raises_enriched_error_without_fallback(monkeypatch):
    class DummyEventBus:
        def __init__(self):
            self.published = []

        def publish(self, topic, payload):
            self.published.append((topic, payload))

    class DummyRegistry:
        class Graph:
            def __init__(self):
                self.nodes = {}

        def __init__(self):
            self.graph = self.Graph()
            self.retry_calls = []
            self.event_bus = DummyEventBus()

        def force_internalization_retry(self, bot_name, delay=0.0):
            self.retry_calls.append((bot_name, delay))

    class TimeoutFuture:
        def result(self, timeout=None):
            raise scm.concurrent.futures.TimeoutError()

    class TimeoutExecutor:
        def __init__(self, max_workers=1):
            self.max_workers = max_workers

        def submit(self, func):
            return TimeoutFuture()

        def shutdown(self, wait=False, cancel_futures=False):
            return None

    registry = DummyRegistry()
    monkeypatch.setattr(scm, "_MANAGER_CONSTRUCTION_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(
        scm,
        "_resolve_manager_timeout_seconds",
        lambda bot_name: 0.01,
    )
    monkeypatch.setattr(
        scm,
        "_resolve_manager_retry_timeout_seconds",
        lambda bot_name, primary_timeout: 0.02,
    )
    monkeypatch.setattr(scm.concurrent.futures, "ThreadPoolExecutor", TimeoutExecutor)

    def _raise_fallback_error(bot_registry, data_bot):
        raise RuntimeError("fallback unavailable")

    monkeypatch.setattr(scm, "_cooldown_disabled_manager", _raise_fallback_error)

    with pytest.raises(RuntimeError, match="fallback unavailable"):
        scm.internalize_coding_bot(
            "BotPlanningBot",
            object(),
            object(),
            data_bot=types.SimpleNamespace(),
            bot_registry=registry,
            provenance_token="token",
        )


def test_internalize_manager_timeout_botplanningbot_retry_exception_raises_underlying_without_fallback(monkeypatch):
    class DummyRegistry:
        class Graph:
            def __init__(self):
                self.nodes = {}

        def __init__(self):
            self.graph = self.Graph()
            self.retry_calls = []
            self.event_bus = None

        def force_internalization_retry(self, bot_name, delay=0.0):
            self.retry_calls.append((bot_name, delay))

    class TimeoutFuture:
        def result(self, timeout=None):
            raise scm.concurrent.futures.TimeoutError()

    class ErrorFuture:
        def result(self, timeout=None):
            raise ValueError("retry blew up")

    class SequencedExecutor:
        calls = 0

        def __init__(self, max_workers=1):
            self.max_workers = max_workers

        def submit(self, func):
            type(self).calls += 1
            if type(self).calls == 1:
                return TimeoutFuture()
            return ErrorFuture()

        def shutdown(self, wait=False, cancel_futures=False):
            return None

    registry = DummyRegistry()
    monkeypatch.setattr(scm, "_MANAGER_CONSTRUCTION_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(
        scm,
        "_resolve_manager_timeout_seconds",
        lambda bot_name: 0.01,
    )
    monkeypatch.setattr(
        scm,
        "_resolve_manager_retry_timeout_seconds",
        lambda bot_name, primary_timeout: 0.02,
    )
    monkeypatch.setattr(scm.concurrent.futures, "ThreadPoolExecutor", SequencedExecutor)
    monkeypatch.setattr(scm, "_cooldown_disabled_manager", lambda bot_registry, data_bot: None)

    with pytest.raises(ValueError, match="retry blew up"):
        scm.internalize_coding_bot(
            "BotPlanningBot",
            object(),
            object(),
            data_bot=types.SimpleNamespace(),
            bot_registry=registry,
            provenance_token="token",
        )


def test_internalize_manager_timeout_botplanningbot_raises_timeout_with_timeout_values(monkeypatch):
    class DummyRegistry:
        class Graph:
            def __init__(self):
                self.nodes = {}

        def __init__(self):
            self.graph = self.Graph()
            self.retry_calls = []
            self.event_bus = None

        def force_internalization_retry(self, bot_name, delay=0.0):
            self.retry_calls.append((bot_name, delay))

    class TimeoutFuture:
        def result(self, timeout=None):
            raise scm.concurrent.futures.TimeoutError()

    class TimeoutExecutor:
        def __init__(self, max_workers=1):
            self.max_workers = max_workers

        def submit(self, func):
            return TimeoutFuture()

        def shutdown(self, wait=False, cancel_futures=False):
            return None

    registry = DummyRegistry()
    monkeypatch.setattr(scm, "_MANAGER_CONSTRUCTION_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(
        scm,
        "_resolve_manager_timeout_seconds",
        lambda bot_name: 0.01,
    )
    monkeypatch.setattr(
        scm,
        "_resolve_manager_retry_timeout_seconds",
        lambda bot_name, primary_timeout: 0.02,
    )
    monkeypatch.setattr(scm.concurrent.futures, "ThreadPoolExecutor", TimeoutExecutor)
    monkeypatch.setattr(scm, "_cooldown_disabled_manager", lambda bot_registry, data_bot: None)

    with pytest.raises(TimeoutError, match="BotPlanningBot") as excinfo:
        scm.internalize_coding_bot(
            "BotPlanningBot",
            object(),
            object(),
            data_bot=types.SimpleNamespace(),
            bot_registry=registry,
            provenance_token="token",
        )

    message = str(excinfo.value)
    assert "0.01s" in message
    assert "0.02s" in message


def test_internalize_manager_timeout_botplanningbot_retry_success_records_phase_metrics(monkeypatch):
    class DummyRegistry:
        class Graph:
            def __init__(self):
                self.nodes = {"BotPlanningBot": {}}

        def __init__(self):
            self.graph = self.Graph()
            self.retry_calls = []
            self.event_bus = None

        def force_internalization_retry(self, bot_name, delay=0.0):
            self.retry_calls.append((bot_name, delay))

        def register_bot(self, _bot_name, **_kwargs):
            return None

    class TimeoutFuture:
        def result(self, timeout=None):
            raise scm.concurrent.futures.TimeoutError()

    class SuccessFuture:
        def __init__(self, func):
            self.func = func

        def result(self, timeout=None):
            return self.func()

    class SequencedExecutor:
        calls = 0

        def __init__(self, max_workers=1):
            self.max_workers = max_workers

        def submit(self, func):
            type(self).calls += 1
            if type(self).calls == 1:
                return TimeoutFuture()
            return SuccessFuture(func)

        def shutdown(self, wait=False, cancel_futures=False):
            return None

    class DummyManager:
        def __init__(self, *args, construction_phase_callback=None, **kwargs):
            self.quick_fix = object()
            self.event_bus = object()
            self.logger = logging.getLogger(__name__)
            if construction_phase_callback is not None:
                construction_phase_callback("manager_init:deferred_scope")

        def initialize_deferred_components(self, skip_non_critical=False):
            return None

    registry = DummyRegistry()
    monkeypatch.setattr(scm, "SelfCodingManager", DummyManager)
    monkeypatch.setattr(scm, "_resolve_manager_timeout_seconds", lambda _bot_name: 0.01)
    monkeypatch.setattr(
        scm,
        "_resolve_manager_retry_timeout_seconds",
        lambda _bot_name, primary_timeout: primary_timeout,
    )
    monkeypatch.setattr(scm.concurrent.futures, "ThreadPoolExecutor", SequencedExecutor)

    manager = scm.internalize_coding_bot(
        "BotPlanningBot",
        object(),
        object(),
        data_bot=types.SimpleNamespace(),
        bot_registry=registry,
        provenance_token="token",
    )

    assert isinstance(manager, DummyManager)
    assert "BOTPLANNINGBOT" in registry._manager_phase_duration_metrics
    per_bot_metrics = registry._manager_phase_duration_metrics["BOTPLANNINGBOT"]
    assert "manager_init:enter" in per_bot_metrics
    node_metrics = registry.graph.nodes["BotPlanningBot"]["manager_phase_duration_metrics"]
    assert node_metrics is per_bot_metrics


def test_generate_patch_enforces_objective_guard(monkeypatch, tmp_path):
    class Engine:
        def __init__(self) -> None:
            self.cognition_layer = types.SimpleNamespace(
                context_builder=types.SimpleNamespace(refresh_db_weights=lambda: None)
            )

    mgr = scm.SelfCodingManager(
        Engine(),
        DummyPipeline(),
        bot_name="bot",
        data_bot=DummyDataBot(),
        bot_registry=DummyRegistry(),
        quick_fix=object(),
        evolution_orchestrator=types.SimpleNamespace(
            register_bot=lambda *a, **k: None, provenance_token="token"
        ),
    )
    monkeypatch.setattr(mgr, "_ensure_quick_fix_engine", lambda *_a, **_k: object())

    def reject(_path):
        raise scm.ObjectiveGuardViolation("unsafe_target", details={"path": "reward_dispatcher.py"})

    monkeypatch.setattr(mgr, "_enforce_objective_guard", reject)

    with pytest.raises(scm.ObjectiveGuardViolation):
        mgr.generate_patch(
            str(tmp_path / "reward_dispatcher.py"),
            provenance_token="token",
            context_builder=types.SimpleNamespace(refresh_db_weights=lambda: None),
        )


def test_generate_and_patch_enforces_objective_guard(monkeypatch, tmp_path):
    class Engine:
        def __init__(self) -> None:
            self.cognition_layer = types.SimpleNamespace(
                context_builder=types.SimpleNamespace(refresh_db_weights=lambda: None)
            )

    mgr = scm.SelfCodingManager(
        Engine(),
        DummyPipeline(),
        bot_name="bot",
        data_bot=DummyDataBot(),
        bot_registry=DummyRegistry(),
        quick_fix=object(),
        evolution_orchestrator=types.SimpleNamespace(
            register_bot=lambda *a, **k: None, provenance_token="token"
        ),
    )

    def reject(_path):
        raise scm.ObjectiveGuardViolation("unsafe_target", details={"path": "billing/stripe_ledger.py"})

    monkeypatch.setattr(mgr, "_enforce_objective_guard", reject)

    with pytest.raises(scm.ObjectiveGuardViolation):
        mgr.generate_and_patch(
            tmp_path / "billing" / "stripe_ledger.py",
            "fix",
            provenance_token="token",
            context_builder=types.SimpleNamespace(refresh_db_weights=lambda: None),
        )


def test_register_patch_cycle_manifest_mismatch_pauses_self_coding(monkeypatch, tmp_path):
    class Engine:
        def __init__(self) -> None:
            self.cognition_layer = types.SimpleNamespace(
                context_builder=types.SimpleNamespace(refresh_db_weights=lambda: None)
            )
            self.patch_db = None

    events: list[tuple[str, dict[str, object]]] = []
    event_bus = types.SimpleNamespace(
        publish=lambda topic, payload: events.append((topic, payload))
    )

    mgr = scm.SelfCodingManager(
        Engine(),
        DummyPipeline(),
        bot_name="bot",
        data_bot=DummyDataBot(),
        bot_registry=DummyRegistry(),
        quick_fix=object(),
        event_bus=event_bus,
        evolution_orchestrator=types.SimpleNamespace(
            register_bot=lambda *a, **k: None, provenance_token="token"
        ),
    )

    def breach() -> None:
        raise scm.ObjectiveGuardViolation(
            "manifest_hash_mismatch",
            details={
                "deltas": [
                    {
                        "path": "reward_dispatcher.py",
                        "expected": "old",
                        "current": "new",
                    }
                ]
            },
        )

    monkeypatch.setattr(mgr.objective_guard, "assert_integrity", breach)

    with pytest.raises(RuntimeError, match="objective integrity breach"):
        mgr.register_patch_cycle("desc", provenance_token="token")

    assert mgr._self_coding_paused is True
    assert any(topic == "self_coding:objective_integrity_breach" for topic, _ in events)

    with pytest.raises(RuntimeError, match="self-coding paused"):
        mgr.register_patch_cycle("desc", provenance_token="token")


def test_reset_objective_integrity_lock_allows_cycles(monkeypatch, tmp_path):
    class Engine:
        def __init__(self) -> None:
            self.cognition_layer = types.SimpleNamespace(
                context_builder=types.SimpleNamespace(refresh_db_weights=lambda: None)
            )
            self.patch_db = None

    mgr = scm.SelfCodingManager(
        Engine(),
        DummyPipeline(),
        bot_name="bot",
        data_bot=DummyDataBot(),
        bot_registry=DummyRegistry(),
        quick_fix=object(),
        evolution_orchestrator=types.SimpleNamespace(
            register_bot=lambda *a, **k: None, provenance_token="token"
        ),
    )

    mgr._self_coding_paused = True
    mgr._self_coding_disabled_reason = "objective_integrity_breach"

    mgr.reset_objective_integrity_lock(operator_id="alice", reason="approved reset after manifest rotation")
    monkeypatch.setattr(mgr, "_enforce_objective_manifest", lambda **_kwargs: None)

    patch_id, commit = mgr.register_patch_cycle("desc", provenance_token="token")

    assert mgr._self_coding_paused is False
    assert mgr._self_coding_disabled_reason is None
    assert patch_id is None
    assert isinstance(commit, str) or commit is None




def test_reset_objective_integrity_lock_requires_manifest_rotation(monkeypatch, tmp_path):
    class Engine:
        def __init__(self) -> None:
            self.cognition_layer = types.SimpleNamespace(
                context_builder=types.SimpleNamespace(refresh_db_weights=lambda: None)
            )
            self.patch_db = None

    mgr = scm.SelfCodingManager(
        Engine(),
        DummyPipeline(),
        bot_name="bot",
        data_bot=DummyDataBot(),
        bot_registry=DummyRegistry(),
        quick_fix=object(),
        evolution_orchestrator=types.SimpleNamespace(
            register_bot=lambda *a, **k: None, provenance_token="token"
        ),
    )
    mgr._self_coding_paused = True
    mgr._self_coding_disabled_reason = "objective_integrity_breach"
    mgr._objective_lock_requires_manifest_refresh = True
    mgr._objective_lock_manifest_sha_at_breach = "same-manifest"

    monkeypatch.setattr(mgr, "_read_manifest_sha", lambda: "same-manifest")

    with pytest.raises(RuntimeError, match="refresh objective hash baseline"):
        mgr.reset_objective_integrity_lock(operator_id="alice", reason="attempt reset")


def test_reset_objective_integrity_lock_clears_pause_after_manifest_rotation(monkeypatch, tmp_path):
    class Engine:
        def __init__(self) -> None:
            self.cognition_layer = types.SimpleNamespace(
                context_builder=types.SimpleNamespace(refresh_db_weights=lambda: None)
            )
            self.patch_db = None

    audit_entries: list[dict[str, object]] = []
    engine = Engine()
    engine.audit_trail = types.SimpleNamespace(record=lambda payload: audit_entries.append(dict(payload)))

    mgr = scm.SelfCodingManager(
        engine,
        DummyPipeline(),
        bot_name="bot",
        data_bot=DummyDataBot(),
        bot_registry=DummyRegistry(),
        quick_fix=object(),
        evolution_orchestrator=types.SimpleNamespace(
            register_bot=lambda *a, **k: None, provenance_token="token"
        ),
    )
    mgr._self_coding_paused = True
    mgr._self_coding_disabled_reason = "objective_integrity_breach"
    mgr._objective_lock_requires_manifest_refresh = True
    mgr._objective_lock_manifest_sha_at_breach = "before"

    monkeypatch.setattr(mgr, "_read_manifest_sha", lambda: "after")

    mgr.reset_objective_integrity_lock(operator_id="alice", reason="rotated objective baseline")

    assert mgr._self_coding_paused is False
    assert mgr._objective_lock_requires_manifest_refresh is False
    assert audit_entries[-1]["event"] == "objective_integrity_lock_reset"

def test_run_post_patch_cycle_circuit_breaker_blocks_followup_patches(monkeypatch, tmp_path):
    class DummySelfTestService:
        def __init__(self, **_kwargs):
            pass

        def run_once(self):
            return (
                {
                    "passed": 0,
                    "failed": 1,
                    "coverage": 0.0,
                    "runtime": 1.0,
                    "stdout": "FAILED tests/test_mod.py::test_feature - AssertionError\n",
                    "module_metrics": {"tests/test_mod.py": {"categories": ["failed"]}},
                },
                [],
            )

    stub_module = types.ModuleType("menace.self_test_service")
    stub_module.SelfTestService = DummySelfTestService
    monkeypatch.setitem(sys.modules, "menace.self_test_service", stub_module)
    monkeypatch.setitem(sys.modules, "self_test_service", stub_module)

    class DummyQuickFix:
        def __init__(self):
            self.context_builder = None
            self.apply_calls = 0

        def validate_patch(self, *_a, **_k):
            return True, []

        def apply_validated_patch(self, *_a, **_k):
            self.apply_calls += 1
            return True, 111, []

    class PersistingRegistry(DummyRegistry):
        def __init__(self, persist_path: Path):
            super().__init__()
            self.persist_path = persist_path
            self.saved = 0

        def save(self, path):
            self.saved += 1
            Path(path).write_text(json.dumps(self.graph.nodes), encoding="utf-8")

    class Engine:
        def __init__(self):
            self.cognition_layer = types.SimpleNamespace(
                context_builder=types.SimpleNamespace(refresh_db_weights=lambda: None)
            )
            self.patch_db = None

    class Pipeline:
        workflow_test_args = ["tests/test_mod.py"]

    events: list[tuple[str, dict[str, object]]] = []
    event_bus = types.SimpleNamespace(
        publish=lambda topic, payload: events.append((topic, dict(payload)))
    )
    quick_fix = DummyQuickFix()
    persist_file = tmp_path / "registry_state.json"
    registry = PersistingRegistry(persist_file)

    monkeypatch.setattr(
        scm,
        "SandboxSettings",
        lambda: types.SimpleNamespace(
            baseline_window=5,
            self_test_repair_retries=1,
            post_patch_repair_attempts=1,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        scm,
        "create_context_builder",
        lambda: types.SimpleNamespace(refresh_db_weights=lambda: None, session_id=None),
    )
    monkeypatch.setattr(scm, "ensure_fresh_weights", lambda builder: None)

    mgr = scm.SelfCodingManager(
        Engine(),
        Pipeline(),
        bot_name="bot",
        data_bot=DummyDataBot(),
        bot_registry=registry,
        quick_fix=quick_fix,
        event_bus=event_bus,
        evolution_orchestrator=types.SimpleNamespace(
            register_bot=lambda *a, **k: None, provenance_token="token"
        ),
    )

    module_path = tmp_path / "module.py"
    module_path.write_text("value = 1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    def fake_run(cmd, *a, **kw):
        if cmd[:2] == ["git", "clone"]:
            dst = Path(cmd[3])
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(module_path, dst / module_path.name)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(scm.subprocess, "run", fake_run)

    rollbacks: list[tuple[str, str]] = []

    class DummyRollbackManager:
        def rollback(self, commit, requesting_bot=None):
            rollbacks.append((str(commit), str(requesting_bot)))

    monkeypatch.setattr(scm, "RollbackManager", DummyRollbackManager)
    mgr._last_commit_hash = "deadbeef"

    checks = {"count": 0}

    def integrity_side_effect():
        checks["count"] += 1
        if checks["count"] >= 4:
            raise scm.ObjectiveGuardViolation(
                "objective_integrity_breach",
                details={"changed_files": ["objective_guard_manifest.json"]},
            )

    monkeypatch.setattr(mgr.objective_guard, "assert_integrity", integrity_side_effect)

    with pytest.raises(RuntimeError, match="circuit breaker triggered"):
        mgr.run_post_patch_cycle(module_path, "initial change", provenance_token="token")

    assert mgr._self_coding_paused is True
    assert mgr._self_coding_disabled_reason == "objective_integrity_breach"
    assert rollbacks == [("deadbeef", "bot")]
    assert quick_fix.apply_calls == 1
    assert any(topic == "self_coding:circuit_breaker_triggered" for topic, _ in events)
    assert any(topic == "self_coding:objective_circuit_breaker_trip" for topic, _ in events)
    assert registry.saved > 0
    assert json.loads(persist_file.read_text(encoding="utf-8"))["bot"][
        "self_coding_disabled"
    ]["reason"] == "objective_integrity_breach"

    mgr2 = scm.SelfCodingManager(
        Engine(),
        Pipeline(),
        bot_name="bot",
        data_bot=DummyDataBot(),
        bot_registry=registry,
        quick_fix=quick_fix,
        evolution_orchestrator=types.SimpleNamespace(
            register_bot=lambda *a, **k: None, provenance_token="token"
        ),
    )
    assert mgr2._self_coding_paused is True
    assert mgr2._self_coding_disabled_reason == "objective_integrity_breach"

def test_run_patch_precommit_objective_mutation_trips_and_restores(monkeypatch, tmp_path):
    class Engine:
        def __init__(self):
            self.cognition_layer = types.SimpleNamespace(
                context_builder=types.SimpleNamespace(refresh_db_weights=lambda: None)
            )
            self.patch_db = None

    class DummyQuickFix:
        def __init__(self, objective_file: Path):
            self.context_builder = None
            self._objective_file = objective_file

        def validate_patch(self, *_a, **_k):
            return True, []

        def apply_validated_patch(self, module_name, *_a, **_k):
            Path(module_name).write_text("value = 2\n", encoding="utf-8")
            self._objective_file.write_text("MUTATED\n", encoding="utf-8")
            return True, 314, []

    class DummyRollbackManager:
        attempts: list[tuple[str, str | None]] = []

        def rollback(self, commit, requesting_bot=None):
            self.attempts.append((str(commit), requesting_bot))

    events: list[tuple[str, dict[str, object]]] = []
    event_bus = types.SimpleNamespace(
        publish=lambda topic, payload: events.append((topic, dict(payload)))
    )

    module_path = tmp_path / "module.py"
    module_path.write_text("value = 1\n", encoding="utf-8")
    protected_file = tmp_path / "reward_dispatcher.py"
    protected_file.write_text("SAFE\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    def fake_run(cmd, *a, **kw):
        if cmd[:2] == ["git", "clone"]:
            dst = Path(cmd[3])
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(module_path, dst / module_path.name)
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd[:3] == ["git", "config", "user.email"] or cmd[:3] == ["git", "config", "user.name"]:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd[:3] == ["git", "checkout", "-b"]:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd[:2] == ["git", "add"]:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd[:2] == ["git", "commit"]:
            raise AssertionError("commit should not run after objective mutation")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(scm.subprocess, "run", fake_run)
    monkeypatch.setattr(
        scm,
        "run_tests",
        lambda *_a, **_k: types.SimpleNamespace(
            success=True, failure=None, stdout="1 passed", stderr="", duration=0.01
        ),
    )
    monkeypatch.setattr(scm, "create_context_builder", lambda: types.SimpleNamespace(refresh_db_weights=lambda: None))
    monkeypatch.setattr(scm, "ensure_fresh_weights", lambda _builder: None)
    monkeypatch.setattr(scm, "RollbackManager", DummyRollbackManager)

    mgr = scm.SelfCodingManager(
        Engine(),
        DummyPipeline(),
        bot_name="bot",
        data_bot=DummyDataBot(),
        bot_registry=DummyRegistry(),
        quick_fix=DummyQuickFix(protected_file),
        event_bus=event_bus,
        evolution_orchestrator=types.SimpleNamespace(
            register_bot=lambda *a, **k: None, provenance_token="token"
        ),
    )
    monkeypatch.setattr(mgr, "_objective_checkpoint_paths", lambda: [protected_file.name])

    with pytest.raises(RuntimeError, match="objective integrity breach"):
        mgr.run_patch(module_path, "mutate objective", provenance_token="token")

    assert mgr._self_coding_paused is True
    assert mgr._self_coding_disabled_reason == "objective_integrity_breach"
    assert protected_file.read_text(encoding="utf-8") == "SAFE\n"
    assert ("314", "bot") in DummyRollbackManager.attempts
    restoration_events = [
        payload
        for topic, payload in events
        if topic == "self_coding:objective_file_mutation_restoration"
    ]
    assert restoration_events
    assert restoration_events[0]["changed_objective_files"] == [protected_file.name]
    assert restoration_events[0]["restoration_ok"] is True
    assert restoration_events[0]["restoration_method"] in {
        "baseline_snapshot",
        "baseline_snapshot+git_checkout",
    }


def test_handle_objective_integrity_breach_emits_trip_event_and_rolls_back(monkeypatch, tmp_path):
    class Engine:
        def __init__(self):
            self.cognition_layer = types.SimpleNamespace(
                context_builder=types.SimpleNamespace(refresh_db_weights=lambda: None)
            )
            self.patch_db = None

    events: list[tuple[str, dict[str, object]]] = []
    event_bus = types.SimpleNamespace(publish=lambda topic, payload: events.append((topic, dict(payload))))

    mgr = scm.SelfCodingManager(
        Engine(),
        DummyPipeline(),
        bot_name="bot",
        data_bot=DummyDataBot(),
        bot_registry=DummyRegistry(),
        quick_fix=object(),
        event_bus=event_bus,
        evolution_orchestrator=types.SimpleNamespace(register_bot=lambda *a, **k: None, provenance_token="token"),
    )
    mgr._last_commit_hash = "abc123"
    mgr._last_patch_id = 7
    rollbacks: list[tuple[str, str]] = []

    class DummyRollbackManager:
        def rollback(self, commit, requesting_bot=None):
            rollbacks.append((str(commit), str(requesting_bot)))

    monkeypatch.setattr(scm, "RollbackManager", DummyRollbackManager)

    with pytest.raises(RuntimeError, match="objective integrity breach"):
        mgr._handle_objective_integrity_breach(
            violation=scm.ObjectiveGuardViolation(
                "objective_integrity_breach",
                details={"changed_files": ["reward_dispatcher.py"]},
            ),
            path=tmp_path / "module.py",
            stage="unit_test",
        )

    assert mgr._self_coding_paused is True
    assert rollbacks == [("abc123", "bot")]
    payload = [p for t, p in events if t == "self_coding:objective_circuit_breaker_trip"][0]
    assert payload["changed_objective_files"] == ["reward_dispatcher.py"]
    assert payload["rollback_ok"] is True


def test_auto_run_patch_objective_breach_trips_circuit_and_rolls_back(monkeypatch, tmp_path):
    class Engine:
        def __init__(self):
            self.cognition_layer = types.SimpleNamespace(
                context_builder=types.SimpleNamespace(refresh_db_weights=lambda: None)
            )
            self.patch_db = None

    events: list[tuple[str, dict[str, object]]] = []
    event_bus = types.SimpleNamespace(
        publish=lambda topic, payload: events.append((topic, dict(payload)))
    )
    audit_entries: list[dict[str, object]] = []
    engine = Engine()
    engine.audit_trail = types.SimpleNamespace(record=lambda payload: audit_entries.append(dict(payload)))

    mgr = scm.SelfCodingManager(
        engine,
        DummyPipeline(),
        bot_name="bot",
        data_bot=DummyDataBot(),
        bot_registry=DummyRegistry(),
        quick_fix=object(),
        event_bus=event_bus,
        evolution_orchestrator=types.SimpleNamespace(
            register_bot=lambda *a, **k: None, provenance_token="token"
        ),
    )
    mgr._last_patch_id = 42

    rollbacks: list[tuple[str, str]] = []

    class DummyRollbackManager:
        def rollback(self, commit, requesting_bot=None):
            rollbacks.append((str(commit), str(requesting_bot)))

    monkeypatch.setattr(scm, "RollbackManager", DummyRollbackManager)

    def raise_drift(*_a, **_k):
        raise scm.ObjectiveGuardViolation(
            "objective_integrity_breach",
            details={"changed_files": ["objective_hash_lock.py"]},
        )

    monkeypatch.setattr(mgr, "run_patch", raise_drift)

    with pytest.raises(RuntimeError, match="objective integrity breach"):
        mgr.auto_run_patch(tmp_path / "module.py", "patch")

    assert mgr._self_coding_paused is True
    assert rollbacks == [("42", "bot")]
    assert any(topic == "self_coding:objective_integrity_trip" for topic, _ in events)
    assert audit_entries[-1]["changed_files"] == ["objective_hash_lock.py"]


def test_idle_cycle_objective_breach_stops_loop_and_skips_followup(monkeypatch, tmp_path):
    class Engine:
        def __init__(self):
            self.cognition_layer = types.SimpleNamespace(
                context_builder=types.SimpleNamespace(refresh_db_weights=lambda: None)
            )
            self.patch_db = None

    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE suggestions (id INTEGER PRIMARY KEY, module TEXT, description TEXT)"
    )
    conn.execute(
        "INSERT INTO suggestions(id, module, description) VALUES (1, ?, 'first')",
        (str(tmp_path / "first.py"),),
    )
    conn.execute(
        "INSERT INTO suggestions(id, module, description) VALUES (2, ?, 'second')",
        (str(tmp_path / "second.py"),),
    )
    conn.commit()

    events: list[tuple[str, dict[str, object]]] = []
    event_bus = types.SimpleNamespace(
        publish=lambda topic, payload: events.append((topic, dict(payload)))
    )
    attempts: list[str] = []

    mgr = scm.SelfCodingManager(
        Engine(),
        DummyPipeline(),
        bot_name="bot",
        data_bot=DummyDataBot(),
        bot_registry=DummyRegistry(),
        quick_fix=object(),
        event_bus=event_bus,
        suggestion_db=types.SimpleNamespace(conn=conn),
        evolution_orchestrator=types.SimpleNamespace(
            register_bot=lambda *a, **k: None, provenance_token="token"
        ),
    )
    mgr._last_patch_id = 99

    rollbacks: list[tuple[str, str]] = []

    class DummyRollbackManager:
        def rollback(self, commit, requesting_bot=None):
            rollbacks.append((str(commit), str(requesting_bot)))

    monkeypatch.setattr(scm, "RollbackManager", DummyRollbackManager)

    def drifting_auto_run(path, description):
        attempts.append(description)
        raise scm.ObjectiveGuardViolation(
            "objective_integrity_breach",
            details={"changed_files": ["config/objective_hash_lock.json"]},
        )

    monkeypatch.setattr(mgr, "auto_run_patch", drifting_auto_run)

    with pytest.raises(RuntimeError, match="objective integrity breach"):
        mgr.idle_cycle()

    assert attempts == ["first"]
    assert mgr._self_coding_paused is True
    assert rollbacks == [("99", "bot")]
    assert any(topic == "self_coding:objective_integrity_trip" for topic, _ in events)

def test_divergence_guard_pauses_when_reward_present_and_real_metrics_missing(monkeypatch):
    class Engine:
        def __init__(self):
            self.patch_db = None
            self.audit_events: list[dict[str, object]] = []
            self.audit_trail = types.SimpleNamespace(
                record=lambda payload: self.audit_events.append(dict(payload))
            )

    events: list[tuple[str, dict[str, object]]] = []
    event_bus = types.SimpleNamespace(
        publish=lambda topic, payload: events.append((topic, dict(payload)))
    )
    engine = Engine()
    mgr = scm.SelfCodingManager(
        engine,
        DummyPipeline(),
        bot_name="bot",
        data_bot=DummyDataBot(),
        bot_registry=DummyRegistry(),
        event_bus=event_bus,
        evolution_orchestrator=types.SimpleNamespace(
            register_bot=lambda *a, **k: None, provenance_token="token"
        ),
    )
    mgr._missing_metric_pause_cycles = 2
    mgr._divergence_fail_closed_on_missing_metrics = True

    mgr._evaluate_divergence_guard({"reward": 1.0, "workflow_id": "wf-1"})
    assert mgr._self_coding_paused is False

    mgr._evaluate_divergence_guard({"reward": 1.2, "workflow_id": "wf-2"})

    assert mgr._self_coding_paused is True
    assert mgr._self_coding_disabled_reason == "reward_real_metrics_unavailable"
    assert mgr._missing_real_metric_streak == 2
    assert any(topic == "self_coding:divergence_kill_switch" for topic, _ in events)
    pause_audit = [
        payload
        for payload in engine.audit_events
        if payload.get("action") == "self_coding_auto_pause"
    ]
    assert pause_audit
    assert pause_audit[-1]["check_status"] == "data_unavailable"


def test_divergence_guard_recovers_after_metrics_resume(monkeypatch):
    class Engine:
        def __init__(self):
            self.patch_db = None
            self.audit_events: list[dict[str, object]] = []
            self.audit_trail = types.SimpleNamespace(
                record=lambda payload: self.audit_events.append(dict(payload))
            )

    events: list[tuple[str, dict[str, object]]] = []
    event_bus = types.SimpleNamespace(
        publish=lambda topic, payload: events.append((topic, dict(payload)))
    )
    engine = Engine()
    mgr = scm.SelfCodingManager(
        engine,
        DummyPipeline(),
        bot_name="bot",
        data_bot=DummyDataBot(),
        bot_registry=DummyRegistry(),
        event_bus=event_bus,
        evolution_orchestrator=types.SimpleNamespace(
            register_bot=lambda *a, **k: None, provenance_token="token"
        ),
    )
    mgr._missing_metric_pause_cycles = 1
    mgr._divergence_fail_closed_on_missing_metrics = True
    mgr._divergence_recovery_cycles = 2

    mgr._evaluate_divergence_guard({"reward": 1.0, "workflow_id": "wf-missing"})
    assert mgr._self_coding_paused is True
    assert mgr._self_coding_pause_source == "divergence_monitor"

    mgr._evaluate_divergence_guard(
        {"reward": 1.1, "profit": 12.0, "revenue": 33.0, "workflow_id": "wf-ok-1"}
    )
    assert mgr._self_coding_paused is True
    assert mgr._missing_real_metric_streak == 0

    mgr._evaluate_divergence_guard(
        {"reward": 1.2, "profit": 12.5, "revenue": 33.5, "workflow_id": "wf-ok-2"}
    )

    assert mgr._self_coding_paused is False
    assert mgr._self_coding_disabled_reason is None
    assert any(topic == "self_coding:divergence_recovered" for topic, _ in events)


def test_internalize_manager_construction_uses_executor_when_timeout_zero(monkeypatch):
    class DummyRegistry:
        class Graph:
            def __init__(self):
                self.nodes = {}

        def __init__(self):
            self.graph = self.Graph()
            self.event_bus = None
            self.modules = {}

        def register_bot(self, _bot_name, **_kwargs):
            return None

    class DummyManager:
        def __init__(self, *args, **kwargs):
            self.quick_fix = object()
            self.event_bus = None
            self.logger = logging.getLogger(__name__)
            self.data_bot = kwargs.get("data_bot")
            self.evolution_orchestrator = None

        def initialize_deferred_components(self, skip_non_critical=False):
            return None

        def run_post_patch_cycle(self, *args, **kwargs):
            return {"self_tests": {"failed": 0}}

    class ImmediateFuture:
        def __init__(self, value):
            self.value = value
            self.requested_timeout = "unset"

        def result(self, timeout=None):
            self.requested_timeout = timeout
            return self.value

    class RecordingExecutor:
        futures: list[ImmediateFuture] = []

        def __init__(self, max_workers=1):
            self.max_workers = max_workers

        def submit(self, func):
            future = ImmediateFuture(func())
            self.futures.append(future)
            return future

        def shutdown(self, wait=False, cancel_futures=False):
            return None

    registry = DummyRegistry()
    registry.graph.nodes["ObserveBot"] = {}

    monkeypatch.setattr(scm, "SelfCodingManager", DummyManager)
    monkeypatch.setattr(scm, "persist_sc_thresholds", lambda *a, **k: None)
    monkeypatch.setattr(scm, "_resolve_manager_timeout_seconds", lambda bot_name: 0.0)
    monkeypatch.setattr(scm.concurrent.futures, "ThreadPoolExecutor", RecordingExecutor)

    manager = scm.internalize_coding_bot(
        "ObserveBot",
        object(),
        object(),
        data_bot=types.SimpleNamespace(),
        bot_registry=registry,
        provenance_token="token",
    )

    assert isinstance(manager, DummyManager)
    assert RecordingExecutor.futures
    assert RecordingExecutor.futures[-1].requested_timeout is None


def test_force_clear_in_flight_entry_emits_structured_log_and_retry(monkeypatch, caplog):
    class DummyRegistry:
        class Graph:
            def __init__(self):
                self.nodes = {}

        def __init__(self):
            self.graph = self.Graph()
            self.retry_calls = []

        def force_internalization_retry(self, bot_name, delay=0.0):
            self.retry_calls.append((bot_name, delay))

    registry = DummyRegistry()
    registry.graph.nodes["WatchdogBot"] = {
        "internalization_last_step": "manager construction",
        "internalization_in_progress": 123.0,
    }
    logger = logging.getLogger("watchdog-test")

    monkeypatch.setattr(scm, "_INTERNALIZE_TIMEOUT_RETRY_BACKOFF_SECONDS", 1.0)
    monkeypatch.setattr(scm, "_INTERNALIZE_TIMEOUT_RETRY_STATE", {})
    monkeypatch.setattr(scm, "_record_internalize_failure", lambda *a, **k: None)
    monkeypatch.setattr(scm, "_record_stale_internalization_failure_event", lambda **k: None)

    with scm._INTERNALIZE_IN_FLIGHT_LOCK:
        scm._INTERNALIZE_IN_FLIGHT["WatchdogBot"] = 10.0

    with caplog.at_level(logging.WARNING):
        scm._force_clear_in_flight_entry(
            bot_name="WatchdogBot",
            started_at=10.0,
            age_seconds=999.0,
            reason="stale_watchdog_force_clear",
            logger=logger,
            bot_registry=registry,
        )

    with scm._INTERNALIZE_IN_FLIGHT_LOCK:
        assert "WatchdogBot" not in scm._INTERNALIZE_IN_FLIGHT
    assert registry.retry_calls and registry.retry_calls[-1][0] == "WatchdogBot"

    matching_logs = [
        rec for rec in caplog.records if getattr(rec, "event", None) == "internalize_in_flight_force_cleared"
    ]
    assert matching_logs
    assert matching_logs[-1].reason == "stale_watchdog_force_clear"
