import subprocess
import sys
import types
from pathlib import Path

import pytest

# -- Stub environment similar to test_self_coding_manager -----------------

pytest.importorskip("networkx")
pytest.importorskip("pandas")

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
    resolve_path=lambda p: Path(p),
    repo_root=lambda: Path("."),
    path_for_prompt=lambda p: str(p),
    resolve_module_path=lambda p: Path(p),
    get_project_root=lambda: Path("."),
)
sys.modules["dynamic_path_router"] = dpr


class _DataBot:
    def roi(self, _bot: str) -> float:
        return 1.0

    def average_errors(self, _bot: str) -> float:
        return 0.0

    def average_test_failures(self, _bot: str) -> float:
        return 0.0

    def get_thresholds(self, _bot: str):
        return types.SimpleNamespace(
            roi_drop=-999.0,
            error_threshold=999.0,
            test_failure_threshold=1.0,
        )

    def check_degradation(self, *_a):
        return True

    def log_evolution_cycle(self, *a, **k):
        pass


db_mod = types.ModuleType("menace.data_bot")
db_mod.DataBot = _DataBot
db_mod.persist_sc_thresholds = lambda *a, **k: None
sys.modules["menace.data_bot"] = db_mod
sys.modules["data_bot"] = db_mod

package = types.ModuleType("menace")
package.__path__ = [str(Path("."))]
sys.modules.setdefault("menace", package)
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
sys.modules["menace.self_coding_engine"] = sce_stub

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
cb_stub = types.ModuleType("coding_bot_interface")
cb_stub.manager_generate_helper = lambda *a, **k: ""
sys.modules.setdefault("coding_bot_interface", cb_stub)
sys.modules.setdefault("menace.coding_bot_interface", cb_stub)

error_logger_stub = types.ModuleType("error_logger")
error_logger_stub.ErrorLogger = object
sys.modules["error_logger"] = error_logger_stub

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
sys.modules["menace.code_database"] = code_db_stub
sys.modules["code_database"] = code_db_stub

ff_stub = types.ModuleType("failure_fingerprint")


class _FF:
    @classmethod
    def from_failure(cls, *a, **k):
        return types.SimpleNamespace()


ff_stub.FailureFingerprint = _FF
sys.modules["failure_fingerprint"] = ff_stub

ffs_stub = types.ModuleType("failure_fingerprint_store")
ffs_stub.FailureFingerprintStore = object
sys.modules["failure_fingerprint_store"] = ffs_stub

ev_stub = types.ModuleType("error_vectorizer")
ev_stub.ErrorVectorizer = object
sys.modules["error_vectorizer"] = ev_stub

vc_stub = types.ModuleType("vector_service.context_builder")
vc_stub.ContextBuilder = object
vc_stub.record_failed_tags = lambda *a, **k: None
vc_stub.load_failed_tags = lambda: set()
vs_mod = types.ModuleType("vector_service")
vs_mod.SharedVectorService = object
vs_mod.CognitionLayer = object
vs_mod.ContextBuilder = object
# Provide ErrorResult so quick_fix_engine import doesn't fail
vs_mod.ErrorResult = Exception
vs_mod.PatchLogger = object
sys.modules["vector_service"] = vs_mod
sys.modules["vector_service.context_builder"] = vc_stub
sys.modules["vector_service.text_preprocessor"] = types.ModuleType(
    "vector_service.text_preprocessor"
)

bot_registry_stub = types.ModuleType("bot_registry")


class _BotRegistry:
    graph = {}

    def register_bot(self, name: str) -> None:
        pass

    def update_bot(self, *a, **k):
        pass

    def save(self, *a, **k):
        pass


bot_registry_stub.BotRegistry = _BotRegistry
sys.modules["bot_registry"] = bot_registry_stub
sys.modules["menace.bot_registry"] = bot_registry_stub

import menace.self_coding_manager as scm  # noqa: E402
import menace.self_coding_thresholds as sct  # noqa: E402

# -- Tests ----------------------------------------------------------------


def test_patch_approval_policy_custom_test_command(monkeypatch, tmp_path):
    calls = {}

    def fake_run(cmd, check=True):
        calls["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(scm.subprocess, "run", fake_run)
    verifier = types.SimpleNamespace(verify=lambda path: True)
    policy = scm.PatchApprovalPolicy(verifier=verifier, test_command=["echo", "ok"])
    assert policy.approve(tmp_path)
    assert calls["cmd"] == ["echo", "ok"]


def test_patch_approval_policy_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("SELF_CODING_TEST_COMMAND", "echo env")
    calls = {}

    def fake_run(cmd, check=True):
        calls["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(scm.subprocess, "run", fake_run)
    verifier = types.SimpleNamespace(verify=lambda path: True)
    policy = scm.PatchApprovalPolicy(verifier=verifier)
    assert policy.approve(tmp_path)
    assert calls["cmd"] == ["echo", "env"]


def test_patch_approval_policy_config_override(monkeypatch, tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("default:\n  test_command: ['echo','cfg']\n")
    monkeypatch.setattr(sct, "_CONFIG_PATH", cfg)
    calls = {}

    def fake_run(cmd, check=True):
        calls["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(scm.subprocess, "run", fake_run)
    verifier = types.SimpleNamespace(verify=lambda path: True)
    policy = scm.PatchApprovalPolicy(verifier=verifier)
    assert policy.approve(tmp_path)
    assert calls["cmd"] == ["echo", "cfg"]

    cfg.write_text(
        """
default:
  test_command: ['echo','cfg']
bots:
  custom:
    test_command: ['echo','bot']
"""
    )
    monkeypatch.setattr(sct, "_CONFIG_PATH", cfg)
    calls.clear()
    policy = scm.PatchApprovalPolicy(verifier=verifier, bot_name="custom")
    assert policy.approve(tmp_path)
    assert calls["cmd"] == ["echo", "bot"]


def test_patch_approval_policy_handles_failure(monkeypatch, tmp_path):
    verifier = types.SimpleNamespace(verify=lambda path: True)

    def fail_run(cmd, check=True):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(scm.subprocess, "run", fail_run)
    policy = scm.PatchApprovalPolicy(verifier=verifier, test_command=["python", "-c", "0/0"])
    assert not policy.approve(tmp_path)


def test_run_patch_custom_clone_command(monkeypatch, tmp_path):
    class DummyEngine:
        def __init__(self):
            builder = types.SimpleNamespace(session_id="", refresh_db_weights=lambda: None)
            self.cognition_layer = types.SimpleNamespace(context_builder=builder)

    class DummyPipeline:
        pass

    class DummyRegistry:
        def register_bot(self, name: str) -> None:
            pass

    class DummyDataBot(_DataBot):
        pass

    engine = DummyEngine()
    pipeline = DummyPipeline()

    class DummyQuickFix:
        def __init__(self, db, mgr, context_builder=None, **kwargs):
            pass
    monkeypatch.setattr(scm, "QuickFixEngine", DummyQuickFix)
    mgr = scm.SelfCodingManager(
        engine,
        pipeline,
        bot_name="bot",
        data_bot=DummyDataBot(),
        bot_registry=DummyRegistry(),
        evolution_orchestrator=types.SimpleNamespace(register_bot=lambda *a, **k: None),
    )

    monkeypatch.setattr(scm, "ensure_fresh_weights", lambda builder: None)

    def _fake_qf():
        mgr.quick_fix = types.SimpleNamespace()
        return mgr.quick_fix

    monkeypatch.setattr(mgr, "_ensure_quick_fix_engine", _fake_qf)

    file_path = tmp_path / "sample.py"
    file_path.write_text("def x():\n    pass\n")
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    commands = []

    def fake_run(cmd, *a, **kw):
        commands.append(cmd)
        raise SystemExit

    monkeypatch.setattr(scm.subprocess, "run", fake_run)
    monkeypatch.setattr(scm.subprocess, "check_output", lambda *a, **k: b"hash")

    with pytest.raises(SystemExit):
        mgr.run_patch(file_path, "add", clone_command=["git", "clone", "--depth", "1"])

    assert commands and commands[0][:4] == ["git", "clone", "--depth", "1"]


def test_run_patch_requires_quick_fix_engine(monkeypatch, tmp_path):
    class DummyEngine:
        def __init__(self):
            builder = types.SimpleNamespace(session_id="", refresh_db_weights=lambda: None)
            self.cognition_layer = types.SimpleNamespace(context_builder=builder)

    class DummyPipeline:
        pass

    class DummyRegistry:
        def register_bot(self, name: str) -> None:
            pass

    class DummyDataBot(_DataBot):
        pass

    engine = DummyEngine()
    pipeline = DummyPipeline()

    class DummyQuickFix:
        def __init__(self, db, mgr, context_builder=None, **kwargs):
            pass

    monkeypatch.setattr(scm, "QuickFixEngine", DummyQuickFix)
    mgr = scm.SelfCodingManager(
        engine,
        pipeline,
        bot_name="bot",
        data_bot=DummyDataBot(),
        bot_registry=DummyRegistry(),
        evolution_orchestrator=types.SimpleNamespace(register_bot=lambda *a, **k: None),
    )

    monkeypatch.setattr(scm, "QuickFixEngine", None)
    mgr.quick_fix = None

    file_path = tmp_path / "sample.py"
    file_path.write_text("def x():\n    pass\n")
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

    with pytest.raises(RuntimeError, match="QuickFixEngine is required"):
        mgr.run_patch(file_path, "add")
