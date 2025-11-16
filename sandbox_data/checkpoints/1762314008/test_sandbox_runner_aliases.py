import importlib
import sys


def _reload_module(name: str) -> None:
    sys.modules.pop(name, None)


def test_cycle_module_aliases():
    """Importing the cycle module via any alias returns the same object."""

    for alias in (
        "sandbox_runner.cycle",
        "menace.sandbox_runner.cycle",
        "menace_sandbox.sandbox_runner.cycle",
    ):
        _reload_module(alias)

    primary = importlib.import_module("menace_sandbox.sandbox_runner.cycle")
    menace_alias = importlib.import_module("menace.sandbox_runner.cycle")
    legacy_alias = importlib.import_module("sandbox_runner.cycle")

    assert primary is menace_alias is legacy_alias
