import os
from pathlib import Path


def test_resolve_path_with_multiple_env_roots(monkeypatch, tmp_path):
    repo_a = tmp_path / "repo_a"
    repo_b = tmp_path / "repo_b"
    for repo in (repo_a, repo_b):
        (repo / ".git").mkdir(parents=True)
    (repo_a / "a.txt").write_text("a")
    (repo_b / "b.txt").write_text("b")

    env_value = os.pathsep.join([str(repo_a), str(repo_b)])
    monkeypatch.setenv("SANDBOX_REPO_PATHS", env_value)

    import dynamic_path_router as dpr
    dpr.clear_cache()

    assert dpr.resolve_path("a.txt") == (repo_a / "a.txt").resolve()
    assert dpr.resolve_path("b.txt", root="repo_b") == (repo_b / "b.txt").resolve()


def test_get_project_root_start_hint(monkeypatch, tmp_path):
    for var in [
        "SANDBOX_REPO_PATHS",
        "MENACE_ROOTS",
        "SANDBOX_REPO_PATH",
        "MENACE_ROOT",
    ]:
        monkeypatch.delenv(var, raising=False)
    repo = tmp_path / "root"
    (repo / ".git").mkdir(parents=True)
    nested = repo / "nested" / "deep"
    nested.mkdir(parents=True)

    import dynamic_path_router as dpr
    dpr.clear_cache()

    assert dpr.get_project_root(start=nested) == repo.resolve()
