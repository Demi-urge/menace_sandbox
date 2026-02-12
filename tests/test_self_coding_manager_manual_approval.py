import json
import logging
import subprocess
import sys
import types
from pathlib import Path

import pytest

from objective_surface_policy import OBJECTIVE_ADJACENT_UNSAFE_PATHS

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

    monkeypatch.setattr(
        scm,
        "is_self_coding_unsafe_path",
        lambda _path, *, repo_root=None: True,
    )
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
    assert policy.last_decision["approval_actor"].startswith("automation")
    assert policy.last_decision["approval_timestamp"]
    assert policy.last_decision["approval_rationale"] == "manual_approval_missing"


def test_non_objective_target_uses_automated_approval(monkeypatch, tmp_path):
    class DummyVerifier:
        def verify(self, path: Path) -> bool:
            return True

    monkeypatch.setattr(
        scm,
        "is_self_coding_unsafe_path",
        lambda _path, *, repo_root=None: False,
    )
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
        last_decision = {
            "approved": False,
            "reason_codes": ("manual_approval_missing",),
            "approval_source": None,
        }

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
    assert any(topic == "self_coding:manual_approval_required" for topic, _ in events)


def test_objective_adjacent_target_allows_manual_approval_artifact(monkeypatch, tmp_path):
    class DummyVerifier:
        def verify(self, path: Path) -> bool:
            return True

    monkeypatch.setattr(
        scm,
        "is_self_coding_unsafe_path",
        lambda _path, *, repo_root=None: True,
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 0),
    )

    policy = scm.PatchApprovalPolicy(verifier=DummyVerifier(), bot_name="bot")
    file_path = tmp_path / "billing" / "stripe_ledger.py"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("x = 1\n", encoding="utf-8")

    artifact = tmp_path / "manual_approval.json"
    artifact.write_text(
        json.dumps({"approved_paths": [str(file_path)]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("MENACE_MANUAL_APPROVAL_FILE", str(artifact))

    assert policy.approve(file_path) is True


def test_run_patch_denied_includes_reason_code():
    events: list[tuple[str, dict[str, object]]] = []

    class DummyApproval:
        last_decision = {
            "approved": False,
            "reason_codes": ("manual_approval_missing",),
            "approval_source": None,
        }

        def classify_target(self, _path: Path) -> str:
            return "objective_adjacent"

        def approve(self, _path: Path, *, manual_approval_token: str | None = None) -> bool:
            return False

    mgr = object.__new__(scm.SelfCodingManager)
    mgr.bot_name = "bot"
    mgr.logger = logging.getLogger("manual-approval-reason-test")
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

    denied = [payload for topic, payload in events if topic == "self_coding:approval_denied"]
    assert denied
    assert denied[0]["reason_code"] == "manual_approval_missing"
    assert denied[0]["reason_codes"] == ("manual_approval_missing",)


def test_manual_approval_policy_blocks_all_canonical_objective_paths(monkeypatch, tmp_path):
    from menace_sandbox.self_coding_policy import is_self_coding_unsafe_path as real_is_unsafe

    class DummyVerifier:
        def verify(self, path: Path) -> bool:
            return True

    monkeypatch.setattr(
        scm,
        "is_self_coding_unsafe_path",
        lambda path, *, repo_root=None: real_is_unsafe(path, repo_root=tmp_path),
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0], 0),
    )

    policy = scm.PatchApprovalPolicy(verifier=DummyVerifier(), bot_name="bot")

    for rule in OBJECTIVE_ADJACENT_UNSAFE_PATHS:
        if rule.endswith("/"):
            candidate = tmp_path / rule.rstrip("/") / "nested.py"
        else:
            candidate = tmp_path / rule
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_text("x = 1\n", encoding="utf-8")

        assert policy.approve(candidate) is False, rule
        assert "manual_approval_missing" in policy.last_decision["reason_codes"]
