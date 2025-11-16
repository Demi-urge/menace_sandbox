import importlib
import sys
import builtins

import pytest


def _reload_backend(monkeypatch, missing):
    """Reload menace.numeric_backend with selected modules missing."""
    for mod in ["menace.numeric_backend", "numeric_backend", "menace"]:
        sys.modules.pop(mod, None)
    orig_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in missing:
            raise ImportError(f"mocked missing module: {name}")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    return importlib.import_module("menace.numeric_backend")


def test_backend_detects_present():
    import menace.numeric_backend as nb
    assert nb.NUMERIC_BACKEND in {"torch", "numpy"}


def test_backend_falls_back_to_numpy(monkeypatch):
    nb = _reload_backend(monkeypatch, {"torch"})
    assert nb.NUMERIC_BACKEND == "numpy"


def test_backend_missing_raises(monkeypatch):
    with pytest.raises(ImportError):
        _reload_backend(monkeypatch, {"torch", "numpy"})
