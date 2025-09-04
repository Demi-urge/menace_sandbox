import shutil
from pathlib import Path


# Helper to create a minimal repo copy with selected files
def _make_repo(tmp_path: Path, *, nested: bool) -> Path:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)

    src_root = Path(__file__).resolve().parents[1]
    shutil.copy2(src_root / "sandbox_runner.py", repo / "sandbox_runner.py")

    if nested:
        dst_dir = repo / "deep" / "nested"
        dst_dir.mkdir(parents=True)
        shutil.copy2(src_root / "self_coding_engine.py", dst_dir / "self_coding_engine.py")
    else:
        shutil.copy2(src_root / "self_coding_engine.py", repo / "self_coding_engine.py")

    return repo


def test_resolve_path_with_menace_root(monkeypatch, tmp_path):
    repo = _make_repo(tmp_path, nested=False)
    monkeypatch.setenv("MENACE_ROOT", str(repo))
    import menace.dynamic_path_router as dpr
    dpr.clear_cache()
    try:
        resolved = dpr.resolve_path("sandbox_runner.py")
        assert resolved == (repo / "sandbox_runner.py").resolve()
    finally:
        dpr.clear_cache()


def test_resolve_path_with_sandbox_repo_path_walk(monkeypatch, tmp_path):
    repo = _make_repo(tmp_path, nested=True)
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))
    import menace.dynamic_path_router as dpr
    dpr.clear_cache()
    try:
        resolved = dpr.resolve_path("self_coding_engine.py")
        expected = (repo / "deep" / "nested" / "self_coding_engine.py").resolve()
        assert resolved == expected
    finally:
        dpr.clear_cache()
