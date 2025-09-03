import subprocess
from pathlib import Path

import pytest

from dynamic_path_router import clear_cache, repo_root, resolve_path


def test_resolve_path_env_override(tmp_path, monkeypatch):
    target = tmp_path / "sample.txt"
    target.write_text("data")
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    clear_cache()
    assert resolve_path("sample.txt") == target.resolve()


def test_repo_root_uses_git(monkeypatch):
    monkeypatch.delenv("SANDBOX_REPO_PATH", raising=False)
    clear_cache()
    root = repo_root()
    assert (root / ".git").exists()


def test_os_walk_fallback(tmp_path, monkeypatch):
    root = tmp_path / "root"
    nested = root / "a" / "b" / "file.txt"
    nested.parent.mkdir(parents=True)
    nested.write_text("x")
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(root))
    clear_cache()
    assert resolve_path("file.txt") == nested.resolve()


def test_nested_repo_submodule(monkeypatch, tmp_path):
    root = tmp_path / "main"
    sub = root / "submodule"
    file = sub / "inner.txt"
    sub.mkdir(parents=True)
    (sub / ".git").mkdir()
    file.write_text("ok")
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(root))
    clear_cache()
    assert resolve_path("submodule/inner.txt") == file.resolve()
    assert resolve_path("inner.txt") == file.resolve()


def test_unknown_file_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    clear_cache()
    with pytest.raises(FileNotFoundError):
        resolve_path("does_not_exist.txt")
