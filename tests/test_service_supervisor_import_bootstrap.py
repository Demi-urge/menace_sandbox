import importlib
import importlib.util
import sys
import types
from pathlib import Path


class _Dummy:
    def __call__(self, *args, **kwargs):
        return _Dummy()

    def __getattr__(self, _name):
        return _Dummy()


class _DummyBotRegistry:
    def __init__(self, *args, **kwargs):
        pass


class _DummyDataBot:
    def __init__(self, *args, **kwargs):
        pass


def _stub_module(name: str, **attrs: object) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)

    def _fallback(_attr: str):
        return _Dummy()

    mod.__getattr__ = _fallback  # type: ignore[attr-defined]
    return mod


def test_script_mode_imports_never_attempt_relative_paths(monkeypatch):
    """Script-style module loading should use absolute imports only."""
    calls: list[tuple[str, str | None]] = []

    repo_root = Path(__file__).resolve().parents[1]
    service_supervisor_path = repo_root / "service_supervisor.py"

    monkeypatch.setitem(
        sys.modules,
        "bootstrap_timeout_policy",
        _stub_module(
            "bootstrap_timeout_policy",
            _BOOTSTRAP_TIMEOUT_MINIMUMS={"MENACE_BOOTSTRAP_WAIT_SECS": 1.0},
            derive_bootstrap_timeout_env=lambda minimum: {"MENACE_BOOTSTRAP_WAIT_SECS": minimum},
            enforce_bootstrap_timeout_policy=lambda logger=None: {"MENACE_BOOTSTRAP_WAIT_SECS": 1.0},
            guard_bootstrap_wait_env=lambda: None,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "context_builder_util",
        _stub_module("context_builder_util", create_context_builder=lambda: _Dummy()),
    )
    monkeypatch.setitem(
        sys.modules,
        "vector_service",
        _stub_module("vector_service", __path__=[]),
    )
    monkeypatch.setitem(
        sys.modules,
        "vector_service.context_builder",
        _stub_module("vector_service.context_builder", ContextBuilder=type("ContextBuilder", (), {})),
    )


    def _fake_import_module(name: str, package: str | None = None):
        calls.append((name, package))
        if name.startswith("."):
            raise AssertionError(f"relative import attempted in script mode: {name}")
        if name == "db_router":
            return _stub_module("db_router", GLOBAL_ROUTER=None, init_db_router=lambda *a, **k: _Dummy())
        if name == "dynamic_path_router":
            return _stub_module("dynamic_path_router", resolve_path=lambda p: p)
        if name == "logging_utils":
            return _stub_module("logging_utils", log_record=lambda **kwargs: kwargs)
        if name == "bot_registry":
            return _stub_module("bot_registry", BotRegistry=_DummyBotRegistry)
        if name == "data_bot":
            return _stub_module(
                "data_bot",
                DataBot=_DummyDataBot,
                persist_sc_thresholds=lambda *a, **k: None,
            )
        if name == "shared_event_bus":
            return _stub_module("shared_event_bus", event_bus=object())
        if name == "coding_bot_interface":
            return _stub_module(
                "coding_bot_interface",
                _BOOTSTRAP_STATE=object(),
                _bootstrap_dependency_broker=lambda: _Dummy(),
                _current_bootstrap_context=lambda: None,
                self_coding_managed=lambda *a, **k: (lambda cls: cls),
            )
        if name == "self_coding_manager":
            return _stub_module(
                "self_coding_manager",
                PatchApprovalPolicy=_Dummy,
                _manager_generate_helper_with_builder=lambda *a, **k: None,
                internalize_coding_bot=lambda *a, **k: _Dummy(),
            )
        return _stub_module(name)

    monkeypatch.setattr(importlib, "import_module", _fake_import_module)

    spec = importlib.util.spec_from_file_location(
        "service_supervisor_script_mode", service_supervisor_path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module._IMPORT_MODE == "script"
    assert calls
    assert all(not mod_name.startswith(".") for mod_name, _ in calls)
