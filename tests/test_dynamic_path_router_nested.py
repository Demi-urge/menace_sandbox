import importlib.util
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory


def _load_router(path: Path):
    spec = importlib.util.spec_from_file_location("dpr_temp", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def test_resolve_path_in_nested_clone(monkeypatch):
    project_root = Path(__file__).resolve().parents[1]
    with TemporaryDirectory() as td:
        repo = Path(td) / "clone" / "menace"
        (repo / "patches").mkdir(parents=True)
        (repo / "prompts").mkdir()
        (repo / "nested" / "deep").mkdir(parents=True)

        shutil.copy(project_root / "dynamic_path_router.py", repo / "dynamic_path_router.py")  # path-ignore
        shutil.copy(project_root / "sandbox_runner.py", repo / "sandbox_runner.py")  # path-ignore
        shutil.copy(project_root / "patch_provenance.py", repo / "patches" / "patch_provenance.py")  # path-ignore
        shutil.copy(project_root / "prompt_engine.py", repo / "prompts" / "prompt_engine.py")  # path-ignore

        dpr = _load_router(repo / "dynamic_path_router.py")  # path-ignore

        monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))
        dpr.clear_cache()

        monkeypatch.chdir(repo / "nested" / "deep")

        assert dpr.resolve_path("sandbox_runner.py") == (repo / "sandbox_runner.py").resolve()  # path-ignore
        assert dpr.resolve_path("patch_provenance.py") == (  # path-ignore
            repo / "patches" / "patch_provenance.py"  # path-ignore
        ).resolve()
        assert dpr.resolve_path("prompt_engine.py") == (  # path-ignore
            repo / "prompts" / "prompt_engine.py"  # path-ignore
        ).resolve()
        assert dpr.resolve_module_path("sandbox_runner") == (
            repo / "sandbox_runner.py"  # path-ignore
        ).resolve()
        assert dpr.resolve_module_path("patches.patch_provenance") == (
            repo / "patches" / "patch_provenance.py"  # path-ignore
        ).resolve()
        assert dpr.resolve_dir("prompts") == (repo / "prompts").resolve()


def test_resolve_path_with_relocated_root(monkeypatch):
    project_root = Path(__file__).resolve().parents[1]
    with TemporaryDirectory() as td:
        repo = Path(td) / "relocated"
        (repo / ".git").mkdir(parents=True)
        (repo / "patches").mkdir()
        (repo / "prompts").mkdir()

        shutil.copy(project_root / "dynamic_path_router.py", repo / "dynamic_path_router.py")  # path-ignore
        shutil.copy(project_root / "sandbox_runner.py", repo / "sandbox_runner.py")  # path-ignore
        shutil.copy(project_root / "patch_provenance.py", repo / "patches" / "patch_provenance.py")  # path-ignore
        shutil.copy(project_root / "prompt_engine.py", repo / "prompts" / "prompt_engine.py")  # path-ignore

        dpr = _load_router(repo / "dynamic_path_router.py")  # path-ignore
        monkeypatch.setenv("MENACE_ROOT", str(repo))
        dpr.clear_cache()

        monkeypatch.chdir(Path(td))

        assert dpr.resolve_path("sandbox_runner.py") == (repo / "sandbox_runner.py").resolve()  # path-ignore
        assert dpr.resolve_path("patch_provenance.py") == (  # path-ignore
            repo / "patches" / "patch_provenance.py"  # path-ignore
        ).resolve()
        assert dpr.resolve_path("prompt_engine.py") == (  # path-ignore
            repo / "prompts" / "prompt_engine.py"  # path-ignore
        ).resolve()
        assert dpr.resolve_module_path("sandbox_runner") == (
            repo / "sandbox_runner.py"  # path-ignore
        ).resolve()
        assert dpr.resolve_module_path("patches.patch_provenance") == (
            repo / "patches" / "patch_provenance.py"  # path-ignore
        ).resolve()
        assert dpr.resolve_dir("patches") == (repo / "patches").resolve()
