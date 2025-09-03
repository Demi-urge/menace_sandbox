from pathlib import Path
import subprocess

import pytest

from dynamic_path_router import (
    clear_cache,
    project_root,
    resolve_module_path,
    resolve_path,
)

import dynamic_path_router as dpr


def test_project_root_git_present(monkeypatch):
    """When git is available the root should match rev-parse output."""

    monkeypatch.delenv("SANDBOX_REPO_PATH", raising=False)
    clear_cache()
    expected = Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    ).resolve()
    assert project_root() == expected


def test_project_root_git_absent(monkeypatch):
    """If git is unavailable the nearest .git directory should be used."""

    monkeypatch.delenv("SANDBOX_REPO_PATH", raising=False)
    clear_cache()

    def _fail(*args, **kwargs):  # pragma: no cover - simple exception helper
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "check_output", _fail)
    expected = Path(__file__).resolve().parents[1]
    assert project_root() == expected


def test_resolve_path_fallback_search(tmp_path):
    """A basename lookup should fall back to an os.walk search."""

    clear_cache()
    root = project_root()
    temp_dir = root / "tests" / "tmp_dynamic_router"
    temp_dir.mkdir(exist_ok=True)
    target = temp_dir / "unique_target.txt"
    target.write_text("content")
    try:
        assert resolve_path("unique_target.txt") == target.resolve()
    finally:
        target.unlink()
        temp_dir.rmdir()


def test_resolve_module_path():
    clear_cache()
    path = resolve_module_path("dynamic_path_router")
    assert path.name == "dynamic_path_router.py"


# ---------------------------------------------------------------------------
# New tests exercising resolve_path in various repository layouts
# ---------------------------------------------------------------------------


def make_repo(tmp_path: Path, layout: str) -> tuple[Path, Path]:
    """Create a temporary repository with the specified layout.

    Returns the repository root and the expected location of sandbox_runner.py.
    """
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".git").mkdir()  # minimal git dir

    if layout == "standard":
        target = root / "sandbox_runner.py"
    elif layout == "submodule":
        sub = root / "submodule"
        sub.mkdir()
        (sub / ".git").mkdir()
        target = sub / "sandbox_runner.py"
    elif layout == "relocated":
        target = root / "other" / "place" / "sandbox_runner.py"
        target.parent.mkdir(parents=True)
    else:
        raise ValueError(layout)

    target.write_text("print('hi')\n")
    return root, target


def setup_env(monkeypatch, root: Path) -> None:
    """Configure MENACE_ROOT and clear caches."""
    monkeypatch.setenv("MENACE_ROOT", str(root))
    dpr.clear_cache()


def test_standard_layout(tmp_path, monkeypatch):
    root, expected = make_repo(tmp_path, "standard")
    setup_env(monkeypatch, root)
    assert dpr.resolve_path("sandbox_runner.py") == expected


def test_nested_submodule(tmp_path, monkeypatch):
    root, expected = make_repo(tmp_path, "submodule")
    setup_env(monkeypatch, root)
    assert dpr.resolve_path("sandbox_runner.py") == expected


def test_relocated_sandbox_runner(tmp_path, monkeypatch):
    root, expected = make_repo(tmp_path, "relocated")
    setup_env(monkeypatch, root)

    calls = []
    real_walk = dpr.os.walk

    def spy_walk(*args, **kwargs):
        calls.append(args[0])
        return real_walk(*args, **kwargs)

    monkeypatch.setattr(dpr.os, "walk", spy_walk)

    assert dpr.resolve_path("sandbox_runner.py") == expected
    assert dpr.resolve_path("sandbox_runner.py") == expected  # cached
    assert len(calls) == 1


def test_missing_file_raises(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".git").mkdir()
    setup_env(monkeypatch, root)

    with pytest.raises(FileNotFoundError):
        dpr.resolve_path("does_not_exist.py")


def test_cache_clearing(tmp_path, monkeypatch):
    root, expected = make_repo(tmp_path, "standard")
    setup_env(monkeypatch, root)

    dpr.resolve_path("sandbox_runner.py")
    assert dpr.list_files()  # cache populated

    dpr.clear_cache()
    assert dpr.list_files() == {}
    assert dpr._PROJECT_ROOT is None

    assert dpr.resolve_path("sandbox_runner.py") == expected
