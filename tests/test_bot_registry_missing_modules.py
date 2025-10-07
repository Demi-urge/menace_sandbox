"""Regression tests for Windows-specific import error parsing in bot registry."""

from menace_sandbox.bot_registry import _collect_missing_modules


def test_collect_missing_modules_handles_dll_load_without_name():
    err = ImportError(
        "DLL load failed while importing win_ext: The specified module could not be found."
    )
    missing = _collect_missing_modules(err)
    assert "win_ext" in missing


def test_collect_missing_modules_handles_nested_dll_message():
    err = ImportError(
        "cannot import name 'Helper' from 'pkg' (DLL load failed while importing helper_lib: The specified module could not be found.)"
    )
    missing = _collect_missing_modules(err)
    assert "helper_lib" in missing
