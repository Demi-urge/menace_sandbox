from __future__ import annotations

import importlib
import shutil
import sys
import types
from pathlib import Path


def _purge_menace_sandbox_modules() -> None:
    for name in list(sys.modules):
        if name == "menace_sandbox" or name.startswith("menace_sandbox."):
            sys.modules.pop(name, None)


def _purge_bootstrap_modules() -> None:
    for name in list(sys.modules):
        if name in {"environment_bootstrap", "menace"} or name.startswith((
            "menace.environment_bootstrap",
            "menace_sandbox",
        )):
            sys.modules.pop(name, None)


def test_environment_bootstrap_exports_environmentbootstrapper() -> None:
    _purge_bootstrap_modules()
    sys.modules.setdefault("tomllib", types.ModuleType("tomllib"))

    mod = importlib.import_module("menace.environment_bootstrap")
    assert hasattr(mod, "EnvironmentBootstrapper")

    from menace.environment_bootstrap import EnvironmentBootstrapper

    assert EnvironmentBootstrapper is mod.EnvironmentBootstrapper

    root_mod = importlib.import_module("environment_bootstrap")
    assert hasattr(root_mod, "EnvironmentBootstrapper")


def test_environment_bootstrap_import_source_and_installed(tmp_path, monkeypatch) -> None:
    module = importlib.import_module("menace_sandbox.environment_bootstrap")
    assert module is not None

    _purge_menace_sandbox_modules()

    repo_root = Path(__file__).resolve().parents[1]
    install_root = tmp_path / "site-packages"
    shutil.copytree(repo_root / "menace_sandbox", install_root / "menace_sandbox")

    monkeypatch.setattr(
        sys,
        "path",
        [str(install_root)] + [path for path in sys.path if path != str(repo_root)],
    )

    module = importlib.import_module("menace_sandbox.environment_bootstrap")
    assert module is not None
