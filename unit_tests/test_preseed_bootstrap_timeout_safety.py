import importlib
import logging
import sys
import time
import types
from contextlib import nullcontext


def _install_stub_module(name: str, attrs: dict) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module
    return module


def _load_preseed_bootstrap_module():
    # Ensure a lightweight menace_sandbox namespace exists for imports.
    menace_ns = sys.modules.get("menace_sandbox") or types.ModuleType("menace_sandbox")
    menace_ns.__path__ = getattr(menace_ns, "__path__", [])
    sys.modules["menace_sandbox"] = menace_ns

    # Stubs for dependencies used during module import.
    _install_stub_module(
        "lock_utils", {"SandboxLock": type("SandboxLock", (), {}), "LOCK_TIMEOUT": 1.0}
    )
    _install_stub_module("safe_repr", {"summarise_value": lambda value: f"summary:{value}"})
    _install_stub_module("security.secret_redactor", {"redact_dict": lambda data: data})
    _install_stub_module(
        "bootstrap_readiness",
        {
            "READINESS_STAGES": (),
            "build_stage_deadlines": lambda *_args, **_kwargs: {},
            "lagging_optional_components": lambda *_args, **_kwargs: set(),
            "minimal_online": lambda *_args, **_kwargs: (False, set(), set(), False),
            "stage_for_step": lambda *_args, **_kwargs: None,
        },
    )
    _install_stub_module(
        "bootstrap_timeout_policy",
        {
            "SharedTimeoutCoordinator": type(
                "SharedTimeoutCoordinator",
                (),
                {
                    "__init__": lambda self, *_, **__: None,
                    "mark_component_state": lambda *_, **__: None,
                    "allocate_timeout": lambda *_, **__: None,
                },
            ),
            "_host_load_average": lambda: 0.0,
            "_host_load_scale": lambda *_args, **_kwargs: 1.0,
            "build_progress_signal_hook": lambda *_, **__: lambda *_a, **_k: None,
            "collect_timeout_telemetry": lambda *_, **__: {},
            "enforce_bootstrap_timeout_policy": lambda *_, **__: None,
            "load_component_timeout_floors": lambda *_, **__: {},
            "read_bootstrap_heartbeat": lambda *_, **__: {},
            "compute_prepare_pipeline_component_budgets": lambda *_, **__: {},
            "render_prepare_pipeline_timeout_hints": lambda *_args, **_kwargs: [
                "Increase MENACE_BOOTSTRAP_WAIT_SECS=720 or BOOTSTRAP_STEP_TIMEOUT=720 for slower bootstrap hosts.",
                "Vector-heavy pipelines: set MENACE_BOOTSTRAP_VECTOR_WAIT_SECS=900 or BOOTSTRAP_VECTOR_STEP_TIMEOUT=900 to bypass the legacy 30s cap.",
                "Stagger concurrent bootstraps or shrink watched directories to reduce contention during pipeline and vector service startup.",
            ],
            "record_component_budget_violation": lambda *_, **__: None,
            "_PREPARE_PIPELINE_COMPONENT": "prepare_pipeline",
            "_COMPONENT_TIMEOUT_MINIMUMS": {},
            "_DEFERRED_COMPONENT_TIMEOUT_MINIMUMS": {},
        },
    )

    _install_stub_module(
        "menace_sandbox.coding_bot_interface",
        {
            "_BOOTSTRAP_TIMEOUT_FLOOR": 720.0,
            "_BOOTSTRAP_WAIT_TIMEOUT": 720.0,
            "_resolve_bootstrap_wait_timeout": lambda heavy=False: 900.0 if heavy else 720.0,
            "_PREPARE_PIPELINE_WATCHDOG": {"stages": []},
            "_pop_bootstrap_context": lambda *_, **__: None,
            "_push_bootstrap_context": lambda *_, **__: None,
            "fallback_helper_manager": nullcontext,
            "prepare_pipeline_for_bootstrap": lambda *_, **__: None,
        },
    )

    # Minimal stubs for ancillary modules.
    _install_stub_module("menace_sandbox.db_router", {"set_audit_bootstrap_safe_default": lambda *_, **__: None})

    stub_classes = {
        "menace_sandbox.bot_registry": "BotRegistry",
        "menace_sandbox.code_database": "CodeDB",
        "menace_sandbox.context_builder_util": "create_context_builder",
        "menace_sandbox.data_bot": "DataBot",
        "menace_sandbox.menace_memory_manager": "MenaceMemoryManager",
        "menace_sandbox.model_automation_pipeline": "ModelAutomationPipeline",
        "menace_sandbox.self_coding_engine": "SelfCodingEngine",
        "menace_sandbox.self_coding_manager": "SelfCodingManager",
        "menace_sandbox.self_coding_thresholds": "get_thresholds",
        "menace_sandbox.threshold_service": "ThresholdService",
    }

    for module_name, attr_name in stub_classes.items():
        stub = _install_stub_module(module_name, {attr_name: type(attr_name, (), {})})
        if attr_name == "create_context_builder":
            stub.create_context_builder = lambda **_: object()
        if attr_name == "SelfCodingManager":
            stub.internalize_coding_bot = lambda *_, **__: None
        if attr_name == "DataBot":
            stub.persist_sc_thresholds = lambda *_, **__: None

    sys.modules.pop("sandbox.preseed_bootstrap", None)
    module = importlib.import_module("sandbox.preseed_bootstrap")
    return importlib.reload(module)


def test_prepare_timeout_emits_remediation_guidance(capsys, caplog):
    preseed_bootstrap = _load_preseed_bootstrap_module()

    def slow_task():
        time.sleep(0.1)

    deadline = time.monotonic() + 0.02
    with caplog.at_level(logging.WARNING, logger=preseed_bootstrap.LOGGER.name):
        result = preseed_bootstrap._run_with_timeout(
            slow_task,
            timeout=0.01,
            bootstrap_deadline=deadline,
            description="prepare_pipeline_for_bootstrap",
            abort_on_timeout=False,
        )

    time.sleep(0.15)

    captured = capsys.readouterr().out
    combined_output = captured + caplog.text

    assert result is None
    assert "MENACE_BOOTSTRAP_WAIT_SECS=720" in combined_output
    assert "BOOTSTRAP_VECTOR_STEP_TIMEOUT=900" in combined_output
    assert "remaining_global_window" in combined_output
    assert "Stagger concurrent bootstraps" in combined_output


def test_adaptive_timeout_extends_when_slack_available():
    preseed_bootstrap = _load_preseed_bootstrap_module()

    def slow_task():
        time.sleep(0.05)
        return "done"

    deadline = time.monotonic() + 1.0
    result = preseed_bootstrap._run_with_timeout(
        slow_task,
        timeout=0.01,
        bootstrap_deadline=deadline,
        description="prepare_pipeline_for_bootstrap",
        abort_on_timeout=False,
    )

    assert result == "done"


def test_bootstrap_wrappers_export_minimum_floors(monkeypatch):
    monkeypatch.delenv("MENACE_BOOTSTRAP_WAIT_SECS", raising=False)
    monkeypatch.delenv("MENACE_BOOTSTRAP_VECTOR_WAIT_SECS", raising=False)

    preseed_bootstrap = _load_preseed_bootstrap_module()

    resolved = preseed_bootstrap.initialize_bootstrap_wait_env()

    assert resolved["MENACE_BOOTSTRAP_WAIT_SECS"] >= 720
    assert resolved["MENACE_BOOTSTRAP_VECTOR_WAIT_SECS"] >= 900
    assert preseed_bootstrap._default_step_floor(vector_heavy=False) >= 720
    assert preseed_bootstrap._default_step_floor(vector_heavy=True) >= 900
