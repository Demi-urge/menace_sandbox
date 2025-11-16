"""Tests for error_logger dependency in sandbox environment."""

from __future__ import annotations

import builtins
import importlib
import sys

import pytest


def test_missing_error_logger_raises(monkeypatch):
    """Importing environment fails when error_logger is unavailable."""

    original_import = builtins.__import__

    def _imp(name, *args, **kwargs):
        if name == "error_logger":
            raise ImportError("missing dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "error_logger", raising=False)
    monkeypatch.delitem(sys.modules, "sandbox_runner.environment", raising=False)
    monkeypatch.setattr(builtins, "__import__", _imp)

    with pytest.raises(RuntimeError) as exc:
        importlib.import_module("sandbox_runner.environment")

    msg = str(exc.value)
    assert "error_logger" in msg
    assert "install" in msg.lower()
