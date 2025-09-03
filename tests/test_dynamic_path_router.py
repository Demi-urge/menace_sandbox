import subprocess
from pathlib import Path

from dynamic_path_router import clear_cache, repo_root, resolve_path


def test_repo_root_matches_git(monkeypatch):
    monkeypatch.delenv("SANDBOX_REPO_PATH", raising=False)
    clear_cache()
    expected = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()
    ).resolve()
    assert repo_root() == expected


def test_fallback_search(monkeypatch, tmp_path):
    root = tmp_path / "proj"
    nested = root / "a" / "b" / "target.txt"
    nested.parent.mkdir(parents=True)
    nested.write_text("x")
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(root))
    clear_cache()
    assert resolve_path("target.txt") == nested.resolve()


def test_caching(monkeypatch, tmp_path):
    root = tmp_path / "proj"
    nested = root / "dir" / "cached.txt"
    nested.parent.mkdir(parents=True)
    nested.write_text("data")
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(root))
    clear_cache()

    calls = 0

    original_rglob = Path.rglob

    def counting_rglob(self, pattern):
        nonlocal calls
        calls += 1
        return original_rglob(self, pattern)

    monkeypatch.setattr(Path, "rglob", counting_rglob)

    assert resolve_path("cached.txt") == nested.resolve()
    assert calls == 1
    assert resolve_path("cached.txt") == nested.resolve()
    assert calls == 1  # cached, no additional rglob calls

