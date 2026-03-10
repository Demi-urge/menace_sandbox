import importlib
import sys


def test_menace_sandbox_legacy_symbol_exports():
    package = importlib.import_module("menace_sandbox")

    for symbol in ("patch_generator", "sandbox_runner"):
        assert hasattr(package, symbol)
        assert symbol in package.__all__


def test_menace_legacy_symbol_exports():
    sys.modules.pop("menace", None)
    package = importlib.import_module("menace")

    for symbol in ("audit_logger", "self_debugger_sandbox"):
        assert hasattr(package, symbol)
        assert symbol in package.__all__
