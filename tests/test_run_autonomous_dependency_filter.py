"""Tests for platform-specific dependency filtering in ``run_autonomous``."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "run_autonomous.py"  # path-ignore
    spec = importlib.util.spec_from_file_location("run_autonomous", str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_autonomous"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_filter_dependency_errors_skips_windows_only_packages():
    mod = _load_module()
    errors = {
        "python": ["pyroute2", "pylint"],
        "optional": ["PyRoute2", "httpx"],
        "system": ["ffmpeg"],
    }

    filtered, notes = mod._filter_dependency_errors(errors, platform="nt")

    assert filtered["python"] == ["pylint"]
    assert filtered["optional"] == ["httpx"]
    assert filtered["system"] == ["ffmpeg"]
    assert notes == [
        "Skipping Windows-incompatible Python packages: pyroute2"
    ]


def test_filter_dependency_errors_noop_non_windows():
    mod = _load_module()
    errors = {"python": ["pyroute2"], "optional": ["httpx"]}

    filtered, notes = mod._filter_dependency_errors(errors, platform="posix")

    assert filtered == {"python": ["pyroute2"], "optional": ["httpx"]}
    assert notes == []

