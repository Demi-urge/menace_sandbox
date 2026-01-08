from pathlib import Path
import sys

from vector_service import context_builder


def test_load_bootstrap_helper_package_only(monkeypatch) -> None:
    repo_root = Path(context_builder.__file__).resolve().parent.parent
    repo_parent = repo_root.parent
    monkeypatch.setattr(sys, "path", [str(repo_parent)])
    helper = context_builder._load_bootstrap_helper()
    assert callable(helper)
