import builtins
import importlib
import sys

import pytest


class _MissingSelfCoding:
    def __init__(self, original_import):
        self._original_import = original_import

    def __call__(self, name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"menace_sandbox.self_coding_manager", "self_coding_manager"}:
            raise ImportError("self_coding_manager is unavailable")
        return self._original_import(name, globals, locals, fromlist, level)


def _simulate_missing_self_coding(monkeypatch):
    original = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", _MissingSelfCoding(original))
    monkeypatch.delitem(sys.modules, "menace_sandbox.self_coding_manager", raising=False)


def test_debug_loop_service_requires_self_coding_manager(monkeypatch):
    _simulate_missing_self_coding(monkeypatch)
    monkeypatch.delitem(sys.modules, "menace_sandbox.debug_loop_service", raising=False)
    with pytest.raises(ImportError):
        importlib.import_module("menace_sandbox.debug_loop_service")


def test_stripe_watchdog_requires_self_coding_manager(monkeypatch):
    _simulate_missing_self_coding(monkeypatch)
    monkeypatch.delitem(sys.modules, "menace_sandbox.stripe_watchdog", raising=False)
    with pytest.raises(ImportError):
        importlib.import_module("menace_sandbox.stripe_watchdog")
