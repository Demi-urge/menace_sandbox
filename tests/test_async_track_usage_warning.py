import importlib.util
import types
import sys
from pathlib import Path

import pytest


def test_meta_logger_requires_telemetry(monkeypatch):
    pkg = types.ModuleType("sandbox_runner")
    pkg.__path__ = []  # ensure submodules cannot be resolved
    monkeypatch.setitem(sys.modules, "sandbox_runner", pkg)
    monkeypatch.delitem(sys.modules, "sandbox_runner.cycle", raising=False)

    path = Path(__file__).resolve().parent.parent / "sandbox_runner" / "meta_logger.py"
    spec = importlib.util.spec_from_file_location("sandbox_runner.meta_logger", path)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "sandbox_runner.meta_logger", mod)
    with pytest.raises(ImportError):
        spec.loader.exec_module(mod)
