from __future__ import annotations

from pathlib import Path
import types

import sys

import pytest

sandbox_runner_stub = types.ModuleType("sandbox_runner")
sandbox_runner_stub.run_workflow_simulations = lambda *a, **k: {}
sys.modules["sandbox_runner"] = sandbox_runner_stub
env_stub = types.ModuleType("sandbox_runner.environment")
env_stub.is_self_debugger_sandbox_import_failure = lambda *a, **k: False
env_stub.module_name_from_module_not_found = lambda *a, **k: None
sys.modules["sandbox_runner.environment"] = env_stub

import self_coding_engine as sce
import menace.menace_self_debug as menace_self_debug
import menace_sandbox.menace_workflow_self_debug as workflow_self_debug


def _base_pipeline_result() -> dict[str, object]:
    return {
        "validation": {"valid": True},
        "patch_text": "diff --git a/x b/x",
        "roi_delta": {"total": 1.0},
        "modified_source": "x = 2\n",
    }


def _prepare_pipeline(monkeypatch, module):
    monkeypatch.setattr(module, "_SELF_DEBUG_PAUSED", False)
    monkeypatch.setattr(
        module,
        "_validate_menace_patch_text",
        lambda *a, **k: {"valid": True, "context": {"file_paths": []}},
    )
    monkeypatch.setattr(module, "is_self_coding_unsafe_path", lambda *a, **k: False)
    monkeypatch.setattr(
        module,
        "evaluate_patch_promotion",
        lambda **k: types.SimpleNamespace(allowed=True, reasons=(), roi_delta_total=1.0),
    )

    class _Guard:
        def __init__(self, *a, **k):
            pass

        def assert_integrity(self):
            return None

    monkeypatch.setattr(module, "ObjectiveGuard", _Guard)


def test_menace_self_debug_requires_manual_approval(monkeypatch, tmp_path):
    _prepare_pipeline(monkeypatch, menace_self_debug)

    class _DeniedApproval:
        def __init__(self, *a, **k):
            pass

        def approve(self, *a, **k):
            return False

    monkeypatch.setattr(menace_self_debug, "ObjectiveApprovalPolicy", _DeniedApproval)

    source = tmp_path / "objective.py"
    source.write_text("x = 1\n", encoding="utf-8")
    ok = menace_self_debug._apply_pipeline_patch(
        _base_pipeline_result(), source_path=source, repo_root=tmp_path
    )

    assert ok is False
    assert source.read_text(encoding="utf-8") == "x = 1\n"


def test_workflow_self_debug_allows_patch_with_manual_approval(monkeypatch, tmp_path):
    _prepare_pipeline(monkeypatch, workflow_self_debug)

    class _Approved:
        def __init__(self, *a, **k):
            pass

        def approve(self, *a, **k):
            return True

    monkeypatch.setattr(workflow_self_debug, "ObjectiveApprovalPolicy", _Approved)

    source = tmp_path / "objective.py"
    source.write_text("x = 1\n", encoding="utf-8")
    ok = workflow_self_debug._apply_pipeline_patch(
        _base_pipeline_result(), source_path=source, repo_root=tmp_path
    )

    assert ok is True
    assert source.read_text(encoding="utf-8") == "x = 2\n"


def test_self_coding_engine_requires_manual_approval_for_objective_adjacent(monkeypatch, tmp_path):
    engine = object.__new__(sce.SelfCodingEngine)

    class _Policy:
        def __init__(self, *a, **k):
            self.last_decision = {"reason_codes": ("manual_approval_missing",)}

        def approve(self, *_a, **_k):
            return False

    monkeypatch.setattr(
        sce,
        "load_internal",
        lambda name: types.SimpleNamespace(ObjectiveApprovalPolicy=_Policy),
        raising=False,
    )

    path = Path.cwd() / "self_coding_manager.py"

    try:
        engine._ensure_patch_path_approved(path)
        raised = False
    except RuntimeError as exc:
        raised = True
        assert "manual_approval_missing" in str(exc)

    assert raised


def test_self_coding_engine_allows_with_manual_approval(monkeypatch):
    engine = object.__new__(sce.SelfCodingEngine)

    class _Policy:
        def __init__(self, *a, **k):
            self.last_decision = {"reason_codes": ()}

        def approve(self, *_a, **_k):
            return True

    monkeypatch.setattr(
        sce,
        "load_internal",
        lambda name: types.SimpleNamespace(ObjectiveApprovalPolicy=_Policy),
        raising=False,
    )

    path = Path.cwd() / "self_coding_manager.py"
    engine._ensure_patch_path_approved(path)


def test_self_coding_engine_requires_manual_approval_for_canonical_objective_surface_paths(monkeypatch):
    engine = object.__new__(sce.SelfCodingEngine)

    class _Policy:
        def __init__(self, *a, **k):
            self.last_decision = {"reason_codes": ("manual_approval_missing",)}

        def approve(self, *_a, **_k):
            return False

    monkeypatch.setattr(
        sce,
        "load_internal",
        lambda name: types.SimpleNamespace(ObjectiveApprovalPolicy=_Policy),
        raising=False,
    )

    objective_paths = [
        "reward_dispatcher.py",
        "kpi_reward_core.py",
        "reward_sanity_checker.py",
        "kpi_editing_detector.py",
        "mvp_evaluator.py",
        "evaluation_worker.py",
        "evaluation_manager.py",
        "evaluation_service.py",
        "self_evaluation_service.py",
        "model_evaluation_service.py",
        "evaluation_history_db.py",
        "central_evaluation_loop.py",
        "menace/core/evaluator.py",
        "neurosales/neurosales/hierarchical_reward.py",
        "neurosales/neurosales/reward_ledger.py",
        "billing/billing_ledger.py",
        "billing/billing_logger.py",
        "billing/stripe_ledger.py",
        "stripe_billing_router.py",
        "finance_router_bot.py",
        "stripe_watchdog.py",
    ]

    for rel in objective_paths:
        with pytest.raises(RuntimeError, match="manual_approval_missing"):
            engine._ensure_patch_path_approved(Path.cwd() / rel)
