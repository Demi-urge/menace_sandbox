"""Regression tests for the lightweight sandbox settings fallback loader."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path


def _load_flat_module(path: Path, name: str = "sandbox_settings_fallback_flat"):
    """Load *path* as a top-level module to emulate flat script execution."""

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive guard
        raise RuntimeError(f"unable to create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def test_fallback_supports_flat_import(monkeypatch):
    """Fallback module should import without a package parent on Windows."""

    root = Path(__file__).resolve().parents[1]
    module_path = root / "sandbox_settings_fallback.py"

    # Ensure the helper modules can be resolved via flat imports.
    monkeypatch.syspath_prepend(str(root))
    for mod in ["sandbox_settings_fallback_flat", "dynamic_path_router", "stack_dataset_defaults"]:
        sys.modules.pop(mod, None)

    module = _load_flat_module(module_path)

    settings = module.SandboxSettings()

    assert hasattr(settings, "severity_score_map")
    assert settings.severity_score_map
    assert module.USING_SANDBOX_SETTINGS_FALLBACK is True


def test_light_imports_prefers_fallback(monkeypatch):
    """MENACE_LIGHT_IMPORTS should force the fallback settings loader."""

    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    for mod in [
        "sandbox_settings",
        "menace_sandbox.sandbox_settings",
        "sandbox_settings_pydantic",
        "menace_sandbox.sandbox_settings_pydantic",
        "menace_sandbox",
        "menace_sandbox.sandbox_settings_fallback",
    ]:
        sys.modules.pop(mod, None)

    module = importlib.import_module("sandbox_settings")

    assert module.USING_SANDBOX_SETTINGS_FALLBACK is True

