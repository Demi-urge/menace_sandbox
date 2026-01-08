from __future__ import annotations

import importlib
import shutil
import sys
from pathlib import Path


def _purge_menace_sandbox_modules() -> None:
    for name in list(sys.modules):
        if name == "menace_sandbox" or name.startswith("menace_sandbox."):
            sys.modules.pop(name, None)


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
