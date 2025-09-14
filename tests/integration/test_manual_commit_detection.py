import subprocess
import logging
import types
import sys


def _stub_modules() -> None:
    names = {
        "menace_sandbox.error_parser": {
            "FailureCache": object,
            "ErrorReport": object,
            "ErrorParser": object,
        },
        "menace_sandbox.failure_fingerprint_store": {
            "FailureFingerprint": object,
            "FailureFingerprintStore": object,
        },
        "menace_sandbox.failure_retry_utils": {
            "check_similarity_and_warn": lambda *a, **k: None,
            "record_failure": lambda *a, **k: None,
        },
        "vector_service.context_builder": {
            "record_failed_tags": lambda *a, **k: None,
            "load_failed_tags": lambda *a, **k: [],
            "ContextBuilder": object,
        },
        "vector_service.retriever": {
            "Retriever": object,
            "PatchRetriever": object,
            "FallbackResult": object,
        },
        "vector_service.patch_logger": {
            "_VECTOR_RISK": 0,
        },
        "menace_sandbox.sandbox_runner.test_harness": {
            "run_tests": lambda *a, **k: None,
            "TestHarnessResult": object,
        },
        "menace_sandbox.self_coding_engine": {"SelfCodingEngine": object},
        "menace_sandbox.model_automation_pipeline": {
            "ModelAutomationPipeline": object,
            "AutomationResult": object,
        },
        "menace_sandbox.data_bot": {
            "DataBot": object,
            "persist_sc_thresholds": lambda *a, **k: None,
        },
        "menace_sandbox.error_bot": {"ErrorDB": object},
        "menace_sandbox.advanced_error_management": {
            "FormalVerifier": object,
            "AutomatedRollbackManager": object,
        },
        "menace_sandbox.mutation_logger": {
            "log_mutation": lambda *a, **k: None,
            "record_mutation_outcome": lambda *a, **k: None,
        },
        "menace_sandbox.rollback_manager": {"RollbackManager": object},
        "menace_sandbox.self_improvement.baseline_tracker": {
            "BaselineTracker": object,
        },
        "menace_sandbox.self_improvement.target_region": {
            "TargetRegion": object,
        },
        "menace_sandbox.sandbox_settings": {"SandboxSettings": lambda: None},
        "menace_sandbox.patch_attempt_tracker": {"PatchAttemptTracker": object},
        "menace_sandbox.threshold_service": {
            "ThresholdService": object,
            "threshold_service": types.SimpleNamespace(
                get=lambda _n: types.SimpleNamespace(
                    roi_drop=0, error_threshold=0, test_failure_threshold=0
                )
            ),
        },
        "menace_sandbox.quick_fix_engine": {
            "QuickFixEngine": object,
            "QuickFixEngineError": Exception,
            "generate_patch": lambda *a, **k: None,
        },
        "context_builder_util": {"ensure_fresh_weights": lambda *a, **k: None},
        "menace_sandbox.coding_bot_interface": {
            "manager_generate_helper": lambda *a, **k: "",
        },
        "menace_sandbox.patch_suggestion_db": {"PatchSuggestionDB": object},
        "menace_sandbox.bot_registry": {"BotRegistry": object},
        "menace_sandbox.patch_provenance": {
            "record_patch_metadata": lambda *a, **k: None,
            "get_patch_by_commit": lambda _c: None,
        },
        "menace_sandbox.unified_event_bus": {"UnifiedEventBus": object},
        "menace_sandbox.shared_event_bus": {"event_bus": None},
        "menace_sandbox.code_database": {"PatchRecord": object},
    }
    for name, attrs in names.items():
        mod = types.ModuleType(name)
        for attr, value in attrs.items():
            setattr(mod, attr, value)
        sys.modules[name] = mod
    sys.modules["vector_service"] = types.SimpleNamespace(
        context_builder=sys.modules["vector_service.context_builder"]
    )
    # provide parent packages for nested modules
    sys.modules.setdefault("menace_sandbox.self_improvement", types.ModuleType("menace_sandbox.self_improvement"))
    sys.modules.setdefault("menace_sandbox.sandbox_runner", types.ModuleType("menace_sandbox.sandbox_runner"))


class DummyBus:
    def __init__(self):
        self.events = []

    def publish(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))


def _init_repo(repo):
    subprocess.run(["git", "init"], check=True, cwd=repo)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)


def test_manual_commit_detected(tmp_path, monkeypatch):
    _stub_modules()
    import menace_sandbox.self_coding_manager as scm
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)

    (repo / "file.txt").write_text("1\n")
    subprocess.run(["git", "add", "file.txt"], check=True, cwd=repo)
    subprocess.run(["git", "commit", "-m", "init"], check=True, cwd=repo)
    first_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=repo).strip()

    (repo / "file.txt").write_text("2\n")
    subprocess.run(["git", "commit", "-am", "manual"], check=True, cwd=repo)
    second_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=repo).strip()

    metas = {first_commit: {"patch_id": 1, "provenance_token": "tok"}}

    def fake_get(commit):
        return metas.get(commit)

    monkeypatch.setattr(scm, "get_patch_by_commit", fake_get)

    calls = []

    class DummyRollback:
        def rollback(self, commit, **_):
            calls.append(commit)

    monkeypatch.setattr(scm, "RollbackManager", lambda *a, **k: DummyRollback())

    bus = DummyBus()
    mgr = scm.SelfCodingManager.__new__(scm.SelfCodingManager)
    mgr.enhancement_classifier = types.SimpleNamespace(scan_repo=lambda: [])
    mgr.suggestion_db = None
    mgr.engine = types.SimpleNamespace(patch_suggestion_db=None, event_bus=bus)
    mgr.logger = logging.getLogger("test")
    mgr.event_bus = bus

    monkeypatch.chdir(repo)
    mgr.scan_repo()

    assert calls == [second_commit]
    assert ("self_coding:unauthorised_commit", {"commit": second_commit}) in bus.events
