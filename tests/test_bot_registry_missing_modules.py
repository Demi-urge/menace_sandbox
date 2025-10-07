"""Regression tests for Windows-specific import error parsing in bot registry."""

from menace_sandbox.bot_registry import (
    _collect_missing_modules,
    _is_transient_internalization_error,
)


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


def test_collect_missing_modules_detects_circular_import():
    err = ImportError(
        "cannot import name 'TaskValidationBot' from partially initialized module 'menace_sandbox.task_validation_bot' (most likely due to a circular import)"
    )
    missing = _collect_missing_modules(err)
    assert "menace_sandbox.task_validation_bot" in missing


def test_collect_missing_modules_detects_circular_import_without_partial_hint():
    err = ImportError(
        "cannot import name 'TaskValidationBot' from 'menace_sandbox.task_validation_bot' (most likely due to a circular import)"
    )
    missing = _collect_missing_modules(err)
    assert "menace_sandbox.task_validation_bot" in missing


def test_circular_imports_not_treated_as_transient_errors():
    err = ImportError(
        "cannot import name 'TaskValidationBot' from partially initialized module "
        "'menace_sandbox.task_validation_bot' (most likely due to a circular import)"
    )
    assert _is_transient_internalization_error(err) is False


def test_collect_missing_modules_infers_from_windows_path():
    err = ImportError(
        "DLL load failed: The specified module could not be found.",
        name=None,
        path=r"C:\\Python\\Lib\\site-packages\\win32api.pyd",
    )
    missing = _collect_missing_modules(err)
    assert "win32api" in missing


def test_collect_missing_modules_handles_dll_error_without_module():
    err = ImportError(
        'DLL load failed: The specified module could not be found. Error loading "api-ms-win-core-path-l1-1-0.dll"',
        name=None,
    )
    missing = _collect_missing_modules(err)
    assert "api-ms-win-core-path-l1-1-0" in missing


def test_collect_missing_modules_handles_module_not_found_without_name():
    err = ModuleNotFoundError(
        "DLL load failed while importing quick_fix_engine: The specified module could not be found.",
        name=None,
    )
    missing = _collect_missing_modules(err)
    assert "quick_fix_engine" in missing


def test_transient_detection_with_inferred_missing_module():
    err = ModuleNotFoundError(
        "DLL load failed while importing quick_fix_engine: The specified module could not be found.",
        name=None,
    )
    assert _is_transient_internalization_error(err) is False


def test_transient_detection_with_named_module_reports_non_transient():
    err = ModuleNotFoundError(
        "No module named 'quick_fix_engine'",
        name="quick_fix_engine",
    )
    assert _is_transient_internalization_error(err) is False


def test_collect_missing_modules_handles_procedure_not_found():
    err = ImportError(
        "DLL load failed while importing quick_fix_engine: The specified procedure could not be found.",
        name="quick_fix_engine",
    )
    missing = _collect_missing_modules(err)
    assert "quick_fix_engine" in missing
