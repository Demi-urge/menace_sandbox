import importlib.util
import os
import shutil
from pathlib import Path

import pytest


def test_resolve_path_in_nested_clone(monkeypatch, tmp_path):
    project_root = Path(__file__).resolve().parents[1]

    repo = tmp_path / "clone" / "menace"
    (repo / "patches").mkdir(parents=True)
    (repo / "prompts").mkdir()
    (repo / "nested" / "deep").mkdir(parents=True)

    shutil.copy(project_root / "dynamic_path_router.py", repo / "dynamic_path_router.py")  # path-ignore
    shutil.copy(project_root / "sandbox_runner.py", repo / "sandbox_runner.py")  # path-ignore
    shutil.copy(project_root / "patch_provenance.py", repo / "patches" / "patch_provenance.py")  # path-ignore
    shutil.copy(project_root / "prompt_engine.py", repo / "prompts" / "prompt_engine.py")  # path-ignore

    spec = importlib.util.spec_from_file_location("dynamic_path_router", repo / "dynamic_path_router.py")  # path-ignore
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

    assert dr.resolve_path("sandbox_runner.py") == (repo / "sandbox_runner.py").resolve()  # path-ignore
    assert dr.resolve_path("patch_provenance.py") == (  # path-ignore
        repo / "patches" / "patch_provenance.py"  # path-ignore
    ).resolve()
    assert dr.resolve_path("prompt_engine.py") == (  # path-ignore
        repo / "prompts" / "prompt_engine.py"  # path-ignore
    ).resolve()

    assert calls


def test_get_project_root_and_resolve_path_with_nested_repos(monkeypatch, tmp_path):
    project_root = Path(__file__).resolve().parents[1]

    outer = tmp_path / "outer_repo"
    inner = outer / "inner_repo"
    (outer / ".git").mkdir(parents=True)
    (inner / ".git").mkdir(parents=True)
    (inner / "nested" / "deep").mkdir(parents=True)

    monkeypatch.delenv("SANDBOX_REPO_PATH", raising=False)
    monkeypatch.delenv("MENACE_ROOT", raising=False)

    shutil.copy(project_root / "dynamic_path_router.py", outer / "dynamic_path_router.py")  # path-ignore
    shutil.copy(project_root / "dynamic_path_router.py", inner / "dynamic_path_router.py")  # path-ignore

    (outer / "outer_file.py").write_text("outer")  # path-ignore
    (inner / "inner_file.py").write_text("inner")  # path-ignore

    spec_outer = importlib.util.spec_from_file_location(
        "dynamic_path_router_outer", outer / "dynamic_path_router.py"  # path-ignore
    )
    dr_outer = importlib.util.module_from_spec(spec_outer)
    assert spec_outer.loader
    spec_outer.loader.exec_module(dr_outer)

    spec_inner = importlib.util.spec_from_file_location(
        "dynamic_path_router_inner", inner / "dynamic_path_router.py"  # path-ignore
    )
    dr_inner = importlib.util.module_from_spec(spec_inner)
    assert spec_inner.loader
    spec_inner.loader.exec_module(dr_inner)

    monkeypatch.chdir(inner / "nested" / "deep")

    dr_outer.clear_cache()
    assert dr_outer.get_project_root() == outer.resolve()
    assert dr_outer.resolve_path("outer_file.py") == (outer / "outer_file.py").resolve()  # path-ignore

    dr_inner.clear_cache()
    assert dr_inner.get_project_root() == inner.resolve()
    assert dr_inner.resolve_path("inner_file.py") == (inner / "inner_file.py").resolve()  # path-ignore


def test_multiple_repos_shared_parent(monkeypatch, tmp_path):
    project_root = Path(__file__).resolve().parents[1]

    parent = tmp_path / "parent"
    repo_a = parent / "repo_a"
    repo_b = parent / "repo_b"
    for repo in (repo_a, repo_b):
        (repo / ".git").mkdir(parents=True)
        shutil.copy(project_root / "dynamic_path_router.py", repo / "dynamic_path_router.py")  # path-ignore
        (repo / "target.py").write_text(repo.name)  # path-ignore

    monkeypatch.delenv("SANDBOX_REPO_PATH", raising=False)
    monkeypatch.delenv("MENACE_ROOT", raising=False)

    spec_a = importlib.util.spec_from_file_location(
        "dynamic_path_router_a", repo_a / "dynamic_path_router.py"  # path-ignore
    )
    dr_a = importlib.util.module_from_spec(spec_a)
    assert spec_a.loader
    spec_a.loader.exec_module(dr_a)

    spec_b = importlib.util.spec_from_file_location(
        "dynamic_path_router_b", repo_b / "dynamic_path_router.py"  # path-ignore
    )
    dr_b = importlib.util.module_from_spec(spec_b)
    assert spec_b.loader
    spec_b.loader.exec_module(dr_b)

    monkeypatch.chdir(repo_a)
    dr_a.clear_cache()
    assert dr_a.get_project_root() == repo_a.resolve()
    assert dr_a.resolve_path("target.py") == (repo_a / "target.py").resolve()  # path-ignore
    with pytest.raises(FileNotFoundError):
        dr_a.resolve_path("missing.py")  # path-ignore

    monkeypatch.chdir(repo_b)
    dr_b.clear_cache()
    assert dr_b.get_project_root() == repo_b.resolve()
    assert dr_b.resolve_path("target.py") == (repo_b / "target.py").resolve()  # path-ignore
    with pytest.raises(FileNotFoundError):
        dr_b.resolve_path("missing.py")  # path-ignore
