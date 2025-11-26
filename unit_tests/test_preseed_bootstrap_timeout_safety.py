from __future__ import annotations

import importlib.util
import os
import sys
import time
import types
from pathlib import Path

import pytest


def _stub_module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _load_preseed_bootstrap(monkeypatch):
    repo_root = Path(__file__).resolve().parent.parent
    module_path = repo_root / "sandbox" / "preseed_bootstrap.py"

    stub_pkg = _stub_module("menace_sandbox")
    stub_pkg.__path__ = []
    fallback_manager = types.SimpleNamespace()
    coding_stub = _stub_module(
        "menace_sandbox.coding_bot_interface",
        _BOOTSTRAP_WAIT_TIMEOUT=300.0,
        _resolve_bootstrap_wait_timeout=lambda vector_heavy=False: float(
            os.getenv("BOOTSTRAP_STEP_TIMEOUT", "300.0")
        ),
        _BOOTSTRAP_STATE=types.SimpleNamespace(vector_heavy=False),
        fallback_helper_manager=fallback_manager,
        _pop_bootstrap_context=lambda *_args, **_kwargs: None,
        _push_bootstrap_context=lambda *_args, **_kwargs: None,
        prepare_pipeline_for_bootstrap=lambda *_, **__: None,
    )
    monkeypatch.setitem(sys.modules, "menace_sandbox", stub_pkg)
    monkeypatch.setitem(sys.modules, "menace_sandbox.coding_bot_interface", coding_stub)
    stub_pkg.coding_bot_interface = coding_stub

    stubs: dict[str, types.ModuleType] = {}

    stubs["bot_registry"] = _stub_module(
        "menace_sandbox.bot_registry", BotRegistry=type("BotRegistry", (), {})
    )
    stubs["code_database"] = _stub_module(
        "menace_sandbox.code_database", CodeDB=type("CodeDB", (), {})
    )
    stubs["context_builder_util"] = _stub_module(
        "menace_sandbox.context_builder_util",
        create_context_builder=lambda *_, **__: None,
    )
    stubs["db_router"] = _stub_module(
        "menace_sandbox.db_router", set_audit_bootstrap_safe_default=lambda *_, **__: None
    )
    stubs["data_bot"] = _stub_module(
        "menace_sandbox.data_bot",
        DataBot=type("DataBot", (), {}),
        persist_sc_thresholds=lambda *_, **__: None,
    )
    stubs["menace_memory_manager"] = _stub_module(
        "menace_sandbox.menace_memory_manager",
        MenaceMemoryManager=type("MenaceMemoryManager", (), {}),
    )
    stubs["model_automation_pipeline"] = _stub_module(
        "menace_sandbox.model_automation_pipeline",
        ModelAutomationPipeline=type("ModelAutomationPipeline", (), {}),
    )
    stubs["self_coding_engine"] = _stub_module(
        "menace_sandbox.self_coding_engine",
        SelfCodingEngine=type("SelfCodingEngine", (), {}),
    )
    stubs["self_coding_manager"] = _stub_module(
        "menace_sandbox.self_coding_manager",
        SelfCodingManager=type("SelfCodingManager", (), {}),
        internalize_coding_bot=lambda *_, **__: None,
    )
    stubs["self_coding_thresholds"] = _stub_module(
        "menace_sandbox.self_coding_thresholds",
        get_thresholds=lambda *_, **__: types.SimpleNamespace(
            roi_drop=None, error_increase=None, test_failure_increase=None
        ),
    )
    stubs["threshold_service"] = _stub_module(
        "menace_sandbox.threshold_service",
        ThresholdService=type("ThresholdService", (), {}),
    )

    for name, module in stubs.items():
        setattr(stub_pkg, name, module)
        monkeypatch.setitem(sys.modules, f"menace_sandbox.{name}", module)

    sys.modules.setdefault(
        "safe_repr", _stub_module("safe_repr", summarise_value=lambda value: repr(value))
    )
    sys.modules.setdefault(
        "security.secret_redactor", _stub_module("security.secret_redactor", redact_dict=lambda d: d)
    )

    spec = importlib.util.spec_from_file_location("sandbox.preseed_bootstrap", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_prepare_timeout_floor_escalates_low_override(monkeypatch, caplog):
    monkeypatch.setenv("BOOTSTRAP_STEP_TIMEOUT", "30")
    monkeypatch.setenv("BOOTSTRAP_VECTOR_STEP_TIMEOUT", "30")

    module = _load_preseed_bootstrap(monkeypatch)

    caplog.set_level("WARNING")

    requested_timeout = module._resolve_step_timeout(vector_heavy=False)
    bootstrap_deadline = time.monotonic() + 60
    resolved_timeout = module._resolve_timeout(
        requested_timeout, bootstrap_deadline=bootstrap_deadline, heavy_bootstrap=False
    )
    effective_timeout, timeout_context = module._enforce_prepare_timeout_floor(
        resolved_timeout,
        vector_heavy=False,
        heavy_prepare=False,
        bootstrap_deadline=bootstrap_deadline,
    )

    assert effective_timeout < module._PREPARE_SAFE_TIMEOUT_FLOOR
    assert timeout_context.get("timeout_safe_floor") == module._PREPARE_SAFE_TIMEOUT_FLOOR
    assert any("safe floor" in record.message for record in caplog.records)


def test_prepare_timeout_timeout_logs_remediation_hints(monkeypatch, caplog):
    module = _load_preseed_bootstrap(monkeypatch)

    caplog.set_level("ERROR")

    def _slow_prepare():
        time.sleep(0.05)

    with pytest.raises(TimeoutError):
        module._run_with_timeout(
            _slow_prepare,
            timeout=0.01,
            bootstrap_deadline=None,
            description="prepare_pipeline_for_bootstrap",
            abort_on_timeout=True,
            heavy_bootstrap=False,
            resolved_timeout=(0.01, {"effective_timeout": 0.01, "deadline_remaining": None}),
        )

    timeout_logs = [
        record for record in caplog.records if "timed out after" in record.message
    ]
    assert timeout_logs, "expected a timeout log to be emitted"
    timeout_log = timeout_logs[-1].message
    assert "remediation_hints" in timeout_log
    assert "MENACE_BOOTSTRAP_WAIT_SECS" in timeout_log
