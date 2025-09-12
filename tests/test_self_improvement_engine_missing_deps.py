import importlib
import sys
import builtins
import types
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
menace_stub = types.ModuleType("menace")
menace_stub.__path__ = [str(ROOT)]
sys.modules.setdefault("menace", menace_stub)

ALLOWED = {
    "logging",
    "time",
    "threading",
    "asyncio",
    "os",
    "sys",
    "json",
    "inspect",
    "sqlite3",
    "pickle",
    "io",
    "tempfile",
    "math",
    "shutil",
    "ast",
    "yaml",
    "pathlib",
    "typing",
    "datetime",
    "numpy",
    "socket",
    "contextlib",
    "subprocess",
    "builtins",
    "collections",
    "_io",
}


def _reload_with_missing(monkeypatch, missing):
    for name in missing:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.delitem(sys.modules, "menace.self_improvement", raising=False)
    monkeypatch.delitem(sys.modules, "self_improvement", raising=False)

    orig_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in missing:
            raise ImportError(f"{name} missing")
        top = name.split(".")[0]
        if top in ALLOWED:
            return orig_import(name, globals, locals, fromlist, level)
        mod = types.ModuleType(name)
        if fromlist:
            for attr in fromlist:
                if attr == "LOCAL_TABLES":
                    obj = set()
                elif attr == "radar":
                    obj = types.SimpleNamespace(track=lambda *a, **k: None)
                else:
                    def _factory(*a: object, **k: object) -> types.SimpleNamespace:
                        return types.SimpleNamespace()

                    obj = _factory
                setattr(mod, attr, obj)
        sys.modules[name] = mod
        return mod
    monkeypatch.setattr(builtins, "__import__", _fake_import)
    try:
        return importlib.import_module("menace.self_improvement")
    finally:
        monkeypatch.setattr(builtins, "__import__", orig_import)


def test_missing_sandbox_runner(monkeypatch):
    with pytest.raises(RuntimeError, match="sandbox_runner"):
        _reload_with_missing(
            monkeypatch,
            {"sandbox_runner.environment", "sandbox_runner.orphan_integration"},
        )


def test_missing_quick_fix_engine(monkeypatch):
    mod = _reload_with_missing(monkeypatch, {"quick_fix_engine"})
    with pytest.raises(RuntimeError, match="quick_fix_engine"):
        mod.generate_patch("m", object(), context_builder=object())
