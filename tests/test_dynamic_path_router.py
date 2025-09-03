from pathlib import Path
import subprocess

from dynamic_path_router import (
    clear_cache,
    project_root,
    resolve_module_path,
    resolve_path,
)


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
