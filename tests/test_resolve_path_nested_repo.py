import importlib.util
import os
import shutil
from pathlib import Path


def test_resolve_path_in_nested_clone(monkeypatch, tmp_path):
    project_root = Path(__file__).resolve().parents[1]

    repo = tmp_path / "clone" / "menace"
    (repo / "patches").mkdir(parents=True)
    (repo / "prompts").mkdir()
    (repo / "nested" / "deep").mkdir(parents=True)

    shutil.copy(project_root / "dynamic_path_router.py", repo / "dynamic_path_router.py")
    shutil.copy(project_root / "sandbox_runner.py", repo / "sandbox_runner.py")
    shutil.copy(project_root / "patch_provenance.py", repo / "patches" / "patch_provenance.py")
    shutil.copy(project_root / "prompt_engine.py", repo / "prompts" / "prompt_engine.py")

    spec = importlib.util.spec_from_file_location("dynamic_path_router", repo / "dynamic_path_router.py")
    dr = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(dr)

    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))
    dr.clear_cache()

    monkeypatch.chdir(repo / "nested" / "deep")

    calls = []
    original_walk = os.walk

    def tracking_walk(*args, **kwargs):
        calls.append(args[0])
        return original_walk(*args, **kwargs)

    monkeypatch.setattr(os, "walk", tracking_walk)

    assert dr.resolve_path("sandbox_runner.py") == (repo / "sandbox_runner.py").resolve()
    assert dr.resolve_path("patch_provenance.py") == (
        repo / "patches" / "patch_provenance.py"
    ).resolve()
    assert dr.resolve_path("prompt_engine.py") == (
        repo / "prompts" / "prompt_engine.py"
    ).resolve()

    assert calls
