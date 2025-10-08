"""Regression tests for Windows-specific import error parsing in bot registry."""

from menace_sandbox.bot_registry import (
    SelfCodingUnavailableError,
    _collect_missing_modules,
    _collect_missing_resources,
    _is_probable_filesystem_path,
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


def test_collect_missing_modules_handles_import_halted_message():
    err = ImportError("import of 'menace_sandbox.future_prediction_bots' halted; None")
    missing = _collect_missing_modules(err)
    assert missing == {
        "future_prediction_bots",
        "menace_sandbox",
        "menace_sandbox.future_prediction_bots",
    }


def test_collect_missing_modules_honours_self_coding_unavailable_error():
    err = SelfCodingUnavailableError(
        "self-coding bootstrap failed",
        missing=(
            "menace_sandbox.task_validation_bot",
            "quick_fix_engine",
            "  sklearn.feature_extraction.text  ",
            None,
        ),
    )
    missing = _collect_missing_modules(err)
    assert missing == {
        "menace_sandbox.task_validation_bot",
        "quick_fix_engine",
        "sklearn.feature_extraction.text",
    }


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


def test_collect_missing_modules_uses_windows_path_hint():
    err = ModuleNotFoundError(
        "module import failed", name=None, path=r"C:\\bots\\future_profitability_bot.py"
    )
    missing = _collect_missing_modules(err)
    assert "future_profitability_bot" in missing


def test_collect_missing_modules_handles_constructor_hint():
    err = ImportError(
        "Could not find module 'C:\\menace\\quick_fix_engine.cp311-win_amd64.pyd' "
        "(or one of its dependencies). Try using the full path with constructor syntax.",
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


def test_collect_missing_modules_detects_quick_fix_engine_hint():
    err = ImportError("QuickFixEngine is required but could not be imported")
    missing = _collect_missing_modules(err)
    assert "quick_fix_engine" in missing


def test_collect_missing_modules_detects_quick_fix_engine_install_hint():
    err = ImportError(
        "QuickFixEngine is required but could not be imported; pip install menace[quickfix]"
    )
    missing = _collect_missing_modules(err)
    assert "quick_fix_engine" in missing


def test_collect_missing_modules_handles_runtime_quick_fix_failure():
    err = RuntimeError(
        "context_builder_util helpers are required for quick_fix_engine"
    )
    missing = _collect_missing_modules(err)
    assert "quick_fix_engine" in missing
    assert "context_builder_util" in missing


def test_transient_detection_uses_windows_path_hint():
    err = ModuleNotFoundError(
        "module import failed", name=None, path=r"C:\\bots\\future_lucrativity_bot.py"
    )
    assert _is_transient_internalization_error(err) is False


def test_is_probable_filesystem_path_detects_windows_drive(tmp_path):
    module_file = tmp_path / "bots" / "task_validation_bot.py"
    module_file.parent.mkdir(parents=True, exist_ok=True)
    module_file.write_text("BOT = True", encoding="utf-8")
    windows_style_path = str(module_file).replace("/", "\\")
    assert _is_probable_filesystem_path(windows_style_path) is True


def test_is_probable_filesystem_path_detects_unc_prefix():
    unc_path = r"\\\shared\bots\future_profitability_bot.py"
    assert _is_probable_filesystem_path(unc_path) is True


def test_is_probable_filesystem_path_rejects_module_names():
    assert _is_probable_filesystem_path("menace_sandbox.task_validation_bot") is False


def test_collect_missing_resources_parses_context_builder_hint():
    err = FileNotFoundError(
        "Missing required context builder database(s): bots.db, errors.db"
    )
    missing = _collect_missing_resources(err)
    assert {"bots.db", "errors.db"}.issubset(missing)


def test_collect_missing_resources_uses_filename_attributes():
    err = FileNotFoundError(2, "No such file", "C:/sandbox/data/code.db")
    missing = _collect_missing_resources(err)
    assert "code.db" in missing
