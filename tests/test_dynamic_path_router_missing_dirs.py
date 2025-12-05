import importlib
from pathlib import Path

import pytest

import dynamic_path_router as dpr


@pytest.fixture(autouse=True)
def _reset_cache():
    dpr.clear_cache()
    yield
    dpr.clear_cache()


def test_resolve_path_allows_missing_nested_directories(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    monkeypatch.setenv("MENACE_ROOT", str(repo))

    dpr.clear_cache()

    missing_dir = Path("new") / "nested" / "dir"

    resolved = dpr.resolve_path(missing_dir, allow_missing_parents=True)

    assert resolved == (repo / missing_dir).resolve(strict=False)
    assert not resolved.exists()

    resolved.mkdir(parents=True)
    assert resolved.is_dir()


def test_resolve_path_missing_nested_dir_still_raises_without_flag(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    monkeypatch.setenv("MENACE_ROOT", str(repo))
    monkeypatch.setenv("PATH_RESOLUTION_BOOTSTRAP", "1")

    dpr.clear_cache()

    with pytest.raises(FileNotFoundError):
        dpr.resolve_path("new/nested/dir")
