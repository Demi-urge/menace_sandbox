from __future__ import annotations

import importlib.abc
import importlib.util
import sys
from pathlib import Path

import pytest


class _BlockSklearn(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: object | None, target: object | None = None):
        if fullname.startswith("sklearn"):
            raise ImportError("blocked sklearn for test")
        return None


def test_roi_tracker_import_error_when_sklearn_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ROI_TRACKER_ALLOW_SKLEARN_FALLBACK", raising=False)
    for name in list(sys.modules):
        if name.startswith("sklearn"):
            sys.modules.pop(name)

    blocker = _BlockSklearn()
    monkeypatch.setattr(sys, "meta_path", [blocker] + sys.meta_path)

    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "roi_tracker.py"
    spec = importlib.util.spec_from_file_location("roi_tracker_test", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)

    with pytest.raises(ImportError) as excinfo:
        spec.loader.exec_module(module)

    message = str(excinfo.value)
    assert "pip install scikit-learn" in message
    local_sklearn = repo_root / "sklearn"
    if local_sklearn.is_dir():
        assert "local 'sklearn' directory" in message
