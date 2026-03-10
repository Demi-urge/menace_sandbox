from __future__ import annotations

import importlib
import os
import sys


def _reload_service_module():
    module_name = "menace_sandbox.quick_fix_engine.quick_fix_engine_service"
    if module_name in sys.modules:
        del sys.modules[module_name]
    return importlib.import_module(module_name)


def test_import_does_not_write_marker_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("QUICK_FIX_SHIM_MARKER_PATH", raising=False)

    _reload_service_module()

    assert not (tmp_path / "shim_was_loaded.txt").exists()


def test_start_writes_opt_in_marker(tmp_path, monkeypatch):
    marker = tmp_path / "state" / "shim_was_loaded.txt"
    monkeypatch.setenv("QUICK_FIX_SHIM_MARKER_PATH", os.fspath(marker))

    mod = _reload_service_module()
    mod.start()

    assert marker.exists()
    assert marker.read_text(encoding="utf-8") == "YES"
