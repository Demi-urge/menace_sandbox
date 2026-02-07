"""Minimal sklearn stubs for local usage."""
from __future__ import annotations

from pathlib import Path
import importlib.machinery
import importlib.util
import sys


def _load_real_sklearn() -> object | None:
    local_root = Path(__file__).resolve().parent.parent
    for entry in sys.path:
        try:
            entry_path = Path(entry).resolve()
        except (OSError, RuntimeError):
            continue
        if entry_path == local_root:
            continue
        spec = importlib.machinery.PathFinder.find_spec("sklearn", [entry])
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[__name__] = module
            spec.loader.exec_module(module)
            return module
    return None


_real_sklearn = _load_real_sklearn()

if _real_sklearn is None:
    from . import metrics, model_selection, pipeline, preprocessing
    from .linear_model import LinearRegression

    __all__ = [
        "LinearRegression",
        "metrics",
        "model_selection",
        "pipeline",
        "preprocessing",
    ]
else:
    globals().update(_real_sklearn.__dict__)
