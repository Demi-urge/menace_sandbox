import importlib
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_modules_respect_sandbox_repo_path(tmp_path, monkeypatch):
    sys.modules.pop("dynamic_path_router", None)
    dpr = importlib.import_module("dynamic_path_router")
    clone = tmp_path / "clone"
    shutil.copytree(dpr.get_project_root(), clone)
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(clone))
    dpr.clear_cache()

    for name in [
        "patch_branch_manager.py",  # path-ignore
        "patch_provenance_service.py",  # path-ignore
        "patch_attempt_tracker.py",  # path-ignore
        "sandbox_runner.py",  # path-ignore
        "run_autonomous.py",  # path-ignore
    ]:
        resolved = dpr.resolve_path(name)
        assert str(resolved).startswith(str(clone))

    prompt_path = dpr.path_for_prompt("sandbox_runner.py")
    assert prompt_path.startswith(str(clone))

    pbm_mod = importlib.import_module("patch_branch_manager")
    manager = pbm_mod.PatchBranchManager()
    assert str(manager.repo).startswith(str(clone))
