import importlib
import importlib.util
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from dynamic_path_router import resolve_path


def _load_router(path: Path):
    spec = importlib.util.spec_from_file_location("dpr_temp", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def test_resolve_path_finds_jsonl():
    project_root = Path(__file__).resolve().parents[1]
    dpr = _load_router(project_root / "dynamic_path_router.py")  # path-ignore
    dpr.clear_cache()
    path = dpr.resolve_path("patch_outcomes.jsonl")
    assert path.name == "patch_outcomes.jsonl"
    assert path.exists()


@pytest.mark.parametrize("env_var", ["MENACE_ROOT", "SANDBOX_REPO_PATH"])
def test_resolve_path_with_env_override(monkeypatch, env_var):
    project_root = Path(__file__).resolve().parents[1]
    with TemporaryDirectory() as td:
        repo = Path(td) / "relocated"
        (repo / ".git").mkdir(parents=True)
        shutil.copy(project_root / "dynamic_path_router.py", repo / "dynamic_path_router.py")  # path-ignore
        shutil.copy(
            resolve_path("tests/fixtures/patch_outcomes.jsonl"),
            repo / "patch_outcomes.jsonl",
        )

        dpr_tmp = _load_router(repo / "dynamic_path_router.py")  # path-ignore

        # ensure only the selected env var is set
        for var in {"MENACE_ROOT", "SANDBOX_REPO_PATH"}:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv(env_var, str(repo))

        dpr_tmp.clear_cache()
        monkeypatch.chdir(Path(td))

        assert dpr_tmp.resolve_path("patch_outcomes.jsonl") == (
            repo / "patch_outcomes.jsonl"
        ).resolve()


def test_resolve_path_jsonl_in_nested_repo(monkeypatch):
    project_root = Path(__file__).resolve().parents[1]
    with TemporaryDirectory() as td:
        repo = Path(td) / "clone" / "menace"
        nested = repo / "nested" / "deep"
        nested.mkdir(parents=True)

        shutil.copy(project_root / "dynamic_path_router.py", repo / "dynamic_path_router.py")  # path-ignore
        shutil.copy(
            resolve_path("tests/fixtures/patch_outcomes.jsonl"),
            repo / "patch_outcomes.jsonl",
        )

        dpr_tmp = _load_router(repo / "dynamic_path_router.py")  # path-ignore

        monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))
        dpr_tmp.clear_cache()
        monkeypatch.chdir(nested)

        assert dpr_tmp.resolve_path("patch_outcomes.jsonl") == (
            repo / "patch_outcomes.jsonl"
        ).resolve()
