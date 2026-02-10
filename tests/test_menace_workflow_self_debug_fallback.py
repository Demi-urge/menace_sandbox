from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from dynamic_path_router import resolve_path

sandbox_runner_stub = ModuleType("sandbox_runner")
sandbox_runner_stub.run_workflow_simulations = lambda **_kwargs: object()
sys.modules["sandbox_runner"] = sandbox_runner_stub

sandbox_runner_env_stub = ModuleType("sandbox_runner.environment")
sandbox_runner_env_stub.is_self_debugger_sandbox_import_failure = (
    lambda exc: isinstance(exc, ModuleNotFoundError)
    and getattr(exc, "name", None) in {"menace.self_debugger_sandbox", "self_debugger_sandbox"}
)
sandbox_runner_env_stub.module_name_from_module_not_found = lambda exc: getattr(exc, "name", None)
sys.modules["sandbox_runner.environment"] = sandbox_runner_env_stub

spec = importlib.util.spec_from_file_location(
    "menace_sandbox.menace_workflow_self_debug",
    resolve_path("menace_sandbox/menace_workflow_self_debug.py"),
)
self_debug = importlib.util.module_from_spec(spec)
assert spec and spec.loader is not None
spec.loader.exec_module(self_debug)


class _MetricsLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def log_metrics(self, event_name: str, **payload) -> None:
        self.events.append((event_name, payload))


def _prepare_common(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(self_debug, "_is_self_improvement_certified", lambda: False)
    monkeypatch.setattr(self_debug, "discover_workflow_modules", lambda *_a, **_k: ["demo.workflow"])
    monkeypatch.setattr(self_debug, "SandboxSettings", lambda: SimpleNamespace())
    monkeypatch.setattr(self_debug, "_snapshot_sandbox_settings", lambda *_a, **_k: {"ok": True})
    monkeypatch.setattr(
        self_debug,
        "freeze_cycle",
        lambda **_k: SimpleNamespace(
            payload={"inputs": {"workflow_modules": ["demo.workflow"]}},
            snapshot_id="snap-1",
            path=tmp_path / "snap.json",
        ),
    )
    monkeypatch.setattr(
        self_debug,
        "run_mvp_pipeline",
        lambda *_a, **_k: {"validation": {"valid": True, "flags": []}},
    )
    monkeypatch.setattr(self_debug, "WorkflowDB", lambda *_a, **_k: object())
    monkeypatch.setattr(self_debug, "seed_workflow_db", lambda *_a, **_k: ["seeded"])
    monkeypatch.setattr(self_debug, "create_context_builder", lambda **_k: object())
    monkeypatch.setattr(self_debug, "wrap_with_logging", lambda func, _cfg: func)


def test_run_self_debug_handles_manager_timeout_with_fallback_import_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _prepare_common(monkeypatch, tmp_path)

    metrics_logger = _MetricsLogger()
    metrics_records: list[dict] = []
    monkeypatch.setattr(self_debug, "record_self_debug_metrics", lambda payload: metrics_records.append(payload))

    def _missing_fallback(**_kwargs):
        raise ModuleNotFoundError(
            "No module named 'menace.self_debugger_sandbox'",
            name="menace.self_debugger_sandbox",
        )

    monkeypatch.setattr(self_debug, "run_workflow_simulations", _missing_fallback)

    failure_context = {
        "context_id": "ctx-1",
        "record": {
            "error": "manager construction timed out for self_coding_manager after 3.00s",
            "classification": "timeout",
        },
    }

    with caplog.at_level("ERROR"):
        result = self_debug._run_self_debug(
            repo_root=tmp_path,
            workflow_db_path=tmp_path / "wf.db",
            source_menace_id="menace_self_debug",
            dynamic_workflows=False,
            metrics_logger=metrics_logger,
            correlation_id="corr-1",
            failure_context=failure_context,
        )

    assert result == 2
    assert any(
        r.message == "workflow fallback self-debug dependency/layout failure" for r in caplog.records
    )
    assert any(
        getattr(r, "missing_module", None) == "menace.self_debugger_sandbox" for r in caplog.records
    )
    assert any(
        isinstance(getattr(r, "primary_failure", None), dict)
        and "manager construction timed out" in str(getattr(r, "primary_failure", None))
        for r in caplog.records
    )

    assert metrics_logger.events[-1][1]["exit_reason"] == "fallback_dependency_layout_failure"
    assert metrics_logger.events[-1][1]["classification"] == {"status": "dependency_layout_failure"}
    assert metrics_records[-1]["fallback_failure"]["type"] == "self_debugger_sandbox import failure"
    assert metrics_records[-1]["primary_failure"] == failure_context["record"]


def test_run_self_debug_manager_timeout_context_with_available_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _prepare_common(monkeypatch, tmp_path)

    metrics_logger = _MetricsLogger()
    metrics_records: list[dict] = []
    monkeypatch.setattr(self_debug, "record_self_debug_metrics", lambda payload: metrics_records.append(payload))
    monkeypatch.setattr(self_debug, "run_workflow_simulations", lambda **_kwargs: object())

    failure_context = {
        "context_id": "ctx-2",
        "record": {
            "error": "manager construction timed out for fallback path",
            "classification": "timeout",
        },
    }

    with caplog.at_level("ERROR"):
        result = self_debug._run_self_debug(
            repo_root=tmp_path,
            workflow_db_path=tmp_path / "wf.db",
            source_menace_id="menace_self_debug",
            dynamic_workflows=False,
            metrics_logger=metrics_logger,
            correlation_id="corr-2",
            failure_context=failure_context,
        )

    assert result == 0
    assert all(
        r.message != "workflow fallback self-debug dependency/layout failure" for r in caplog.records
    )
    assert metrics_logger.events[-1][1]["exit_reason"] == "workflow_run_success"
    assert metrics_records[-1]["exit_reason"] == "workflow_run_success"
