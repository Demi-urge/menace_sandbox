from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.machinery
import inspect
import sys
import types
from pathlib import Path


def _seed_packages() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    root_pkg = types.ModuleType("menace_sandbox")
    root_pkg.__path__ = [str(repo_root)]
    root_pkg.__spec__ = importlib.machinery.ModuleSpec(
        "menace_sandbox", loader=None, is_package=True
    )
    sys.modules["menace_sandbox"] = root_pkg

    si_pkg = types.ModuleType("menace_sandbox.self_improvement")
    si_pkg.__path__ = [str(repo_root / "self_improvement")]
    si_pkg.__spec__ = importlib.machinery.ModuleSpec(
        "menace_sandbox.self_improvement", loader=None, is_package=True
    )
    sys.modules["menace_sandbox.self_improvement"] = si_pkg


def test_patch_generation_import_with_minimal_settings_namespace(monkeypatch):
    _seed_packages()

    settings_mod = types.ModuleType("menace_sandbox.sandbox_settings")
    settings_mod.SandboxSettings = lambda: types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "menace_sandbox.sandbox_settings", settings_mod)

    metrics_mod = types.ModuleType("menace_sandbox.metrics_exporter")
    counter = types.SimpleNamespace(labels=lambda **_: counter, inc=lambda: None)
    metrics_mod.self_improvement_failure_total = counter
    monkeypatch.setitem(sys.modules, "menace_sandbox.metrics_exporter", metrics_mod)

    logging_mod = types.ModuleType("menace_sandbox.logging_utils")
    logging_mod.log_record = lambda **fields: fields
    monkeypatch.setitem(sys.modules, "menace_sandbox.logging_utils", logging_mod)

    sys.modules.pop("menace_sandbox.self_improvement.patch_generation", None)
    module = importlib.import_module("menace_sandbox.self_improvement.patch_generation")

    sig = inspect.signature(module.generate_patch)
    assert sig.parameters["retries"].default == 3
    assert sig.parameters["delay"].default == 0.5


@dataclass
class _DataclassSettings:
    patch_retry_delay: float = 0.2


def test_patch_application_import_with_dataclass_settings(monkeypatch):
    _seed_packages()

    settings_mod = types.ModuleType("menace_sandbox.sandbox_settings")
    settings_mod.SandboxSettings = lambda: _DataclassSettings()
    monkeypatch.setitem(sys.modules, "menace_sandbox.sandbox_settings", settings_mod)

    context_mod = types.ModuleType("menace_sandbox.context_builder")
    context_mod.create_context_builder = lambda *a, **k: object()
    monkeypatch.setitem(sys.modules, "menace_sandbox.context_builder", context_mod)

    policy_mod = types.ModuleType("self_coding_policy")
    policy_mod.is_self_coding_unsafe_path = lambda *a, **k: False
    monkeypatch.setitem(sys.modules, "self_coding_policy", policy_mod)

    qf_mod = types.ModuleType("quick_fix_engine")
    qf_mod.quick_fix = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "quick_fix_engine", qf_mod)

    path_mod = types.ModuleType("menace_sandbox.dynamic_path_router")
    path_mod.resolve_path = lambda value: Path(value)
    monkeypatch.setitem(sys.modules, "menace_sandbox.dynamic_path_router", path_mod)

    sys.modules.pop("menace_sandbox.self_improvement.patch_application", None)
    module = importlib.import_module("menace_sandbox.self_improvement.patch_application")

    sig = inspect.signature(module.apply_patch)
    assert sig.parameters["retries"].default == 3
    assert sig.parameters["delay"].default == 0.2
