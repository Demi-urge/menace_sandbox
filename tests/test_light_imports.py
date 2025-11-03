import importlib
import sys

import pytest


def _reload_menace(monkeypatch):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    for name in [
        "menace_sandbox",
        "menace",
        "menace_sandbox.upgrade_forecaster",
        "menace.upgrade_forecaster",
    ]:
        sys.modules.pop(name, None)
    return importlib.import_module("menace_sandbox")


def test_light_imports_skip_heavy_optional_modules(monkeypatch):
    module = _reload_menace(monkeypatch)
    upgrade_cls = module.UpgradeForecaster
    with pytest.raises(RuntimeError) as excinfo:
        upgrade_cls()
    assert "MENACE_LIGHT_IMPORTS" in str(excinfo.value)


def test_light_imports_can_be_disabled(monkeypatch):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "0")
    # Ensure a clean import
    for name in [
        "menace_sandbox",
        "menace",
    ]:
        sys.modules.pop(name, None)

    module = importlib.import_module("menace_sandbox")
    upgrade_cls = module.UpgradeForecaster
    # When light imports are disabled the optional class is exposed directly.
    assert upgrade_cls is not None
