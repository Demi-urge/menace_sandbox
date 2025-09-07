import importlib
import sys

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "menace.auto_escalation_manager",
        "menace.watchdog",
        "menace.enhancement_bot",
        "menace.self_coding_engine",
    ],
)
def test_modules_require_vector_service(module_name, monkeypatch):
    """Modules should fail fast when ``vector_service`` is unavailable."""

    monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.delitem(sys.modules, "vector_service", raising=False)

    with pytest.raises(ImportError):
        importlib.import_module(module_name)
