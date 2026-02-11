import logging
import subprocess
import sys
import types
from pathlib import Path

import pytest

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
    resolve_path=lambda p: Path(p),
    repo_root=lambda: Path("."),
    path_for_prompt=lambda p: str(p),
    get_project_root=lambda: Path("."),
)
sys.modules["dynamic_path_router"] = dpr

import menace.data_bot as db

sys.modules["data_bot"] = db
sys.modules["menace"].RAISE_ERRORS = False

mapl_stub = types.ModuleType("menace.model_automation_pipeline")


class AutomationResult:
    def __init__(self, package=None, roi=None):
        self.package = package
        self.roi = roi


class ModelAutomationPipeline:
    pass


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
qfe_stub.QuickFixEngine = object
qfe_stub.QuickFixEngineError = RuntimeError
qfe_stub.generate_patch = lambda *a, **k: (1, [])
sys.modules.setdefault("quick_fix_engine", qfe_stub)
sys.modules.setdefault("menace.quick_fix_engine", qfe_stub)

th_stub = types.ModuleType("menace.sandbox_runner.test_harness")
th_stub.run_tests = lambda *a, **k: types.SimpleNamespace(
    success=True, failure=None, stdout="", stderr="", duration=0.0
)
th_stub.TestHarnessResult = types.SimpleNamespace
sys.modules.setdefault("menace.sandbox_runner.test_harness", th_stub)

import menace.self_coding_manager as scm


def test_objective_adjacent_target_blocked_without_manual_approval(monkeypatch, tmp_path):
    class DummyVerifier:
        def verify(self, path: Path) -> bool:
            return True

    class _UnsafePolicy:
        def is_safe_target(self, _path: Path) -> bool:
            return False

    monkeypatch.setattr(scm, "get_patch_promotion_policy", lambda repo_root=None: _UnsafePolicy())
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 0),
    )

    policy = scm.PatchApprovalPolicy(verifier=DummyVerifier(), bot_name="bot")
    file_path = tmp_path / "billing" / "stripe_ledger.py"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("x = 1\n", encoding="utf-8")

    assert policy.approve(file_path) is False


def test_non_objective_target_uses_automated_approval(monkeypatch, tmp_path):
    class DummyVerifier:
        def verify(self, path: Path) -> bool:
            return True

    class _SafePolicy:
        def is_safe_target(self, _path: Path) -> bool:
            return True

    monkeypatch.setattr(scm, "get_patch_promotion_policy", lambda repo_root=None: _SafePolicy())
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 0),
    )

    policy = scm.PatchApprovalPolicy(verifier=DummyVerifier(), bot_name="bot")
    file_path = tmp_path / "src" / "utils.py"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("x = 1\n", encoding="utf-8")

    assert policy.approve(file_path) is True


def test_run_patch_emits_required_and_denied_events():
    events: list[tuple[str, dict[str, object]]] = []

    class DummyApproval:
        def classify_target(self, _path: Path) -> str:
            return "objective_adjacent"

        def approve(self, _path: Path, *, manual_approval_token: str | None = None) -> bool:
            return False

    mgr = object.__new__(scm.SelfCodingManager)
    mgr.bot_name = "bot"
    mgr.logger = logging.getLogger("manual-approval-test")
    mgr.event_bus = types.SimpleNamespace(
        publish=lambda topic, payload: events.append((topic, payload))
    )
    mgr.approval_policy = DummyApproval()
    mgr.validate_provenance = lambda _token: None
    mgr._ensure_self_coding_active = lambda: None
    mgr._enforce_objective_guard = lambda _path: None
    mgr.refresh_quick_fix_context = lambda: None

    with pytest.raises(RuntimeError, match="patch approval failed"):
        mgr.run_patch(
            Path("billing/stripe_ledger.py"),
            "desc",
            provenance_token="token",
        )

    assert any(topic == "self_coding:approval_required" for topic, _ in events)
    assert any(topic == "self_coding:approval_denied" for topic, _ in events)
