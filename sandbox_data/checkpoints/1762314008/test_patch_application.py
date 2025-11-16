import sys
import types
import subprocess
import importlib.machinery
from pathlib import Path
import importlib
import importlib.util

import pytest


def _load_patch_module():
    repo_root = Path(__file__).resolve().parents[2]
    root_pkg = types.ModuleType("menace_sandbox")
    root_pkg.__path__ = [str(repo_root)]
    root_pkg.__spec__ = importlib.machinery.ModuleSpec("menace_sandbox", loader=None, is_package=True)
    sys.modules.setdefault("menace_sandbox", root_pkg)
    si_pkg = types.ModuleType("menace_sandbox.self_improvement")
    si_pkg.__path__ = [str(repo_root / "self_improvement")]
    si_pkg.__spec__ = importlib.machinery.ModuleSpec(
        "menace_sandbox.self_improvement", loader=None, is_package=True
    )
    sys.modules.setdefault("menace_sandbox.self_improvement", si_pkg)
    sandbox_settings = types.ModuleType("sandbox_settings")
    class SandboxSettings:
        patch_retries = 1
        patch_retry_delay = 0.0
        meta_entropy_threshold = 0.2
        meta_mutation_rate = 0.0
        meta_roi_weight = 0.0
        meta_domain_penalty = 0.0
        meta_entropy_weight = 0.0
        meta_search_depth = 1
        meta_beam_width = 1
        menace_light_imports = True
    sandbox_settings.SandboxSettings = SandboxSettings
    sandbox_settings.load_sandbox_settings = lambda: SandboxSettings()
    sandbox_runner = types.ModuleType("sandbox_runner")
    bootstrap = types.ModuleType("sandbox_runner.bootstrap")
    bootstrap.initialize_autonomous_sandbox = lambda: None
    sandbox_runner.bootstrap = bootstrap
    sys.modules.setdefault("sandbox_runner", sandbox_runner)
    sys.modules.setdefault("sandbox_runner.bootstrap", bootstrap)
    sys.modules.setdefault("sandbox_settings", sandbox_settings)
    utils = importlib.import_module("menace_sandbox.self_improvement.utils")
    utils.clear_import_cache()
    return importlib.import_module("menace_sandbox.self_improvement.patch_application")


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "you@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Your Name"], cwd=repo, check=True)
    (repo / "file.txt").write_text("hello\n")
    subprocess.run(["git", "add", "file.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)
    return repo

SUCCESS_PATCH = """\
diff --git a/file.txt b/file.txt
--- a/file.txt
+++ b/file.txt
@@ -1 +1 @@
-hello
+world
"""

FAIL_PATCH = """\
diff --git a/file.txt b/file.txt
--- a/file.txt
+++ b/file.txt
@@ -1 +1 @@
-nope
+world
"""

def test_apply_patch_success(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    mod = types.ModuleType("quick_fix_engine")
    mod.fetch_patch = lambda patch_id: SUCCESS_PATCH
    monkeypatch.setitem(sys.modules, "quick_fix_engine", mod)
    patch_module = _load_patch_module()
    commit, diff = patch_module.apply_patch(1, repo)
    assert len(commit) == 40
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True, check=True).stdout.strip()
    assert head == commit
    assert diff == SUCCESS_PATCH
    assert (repo / "file.txt").read_text() == "world\n"

def test_apply_patch_failure(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    head_before = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True, check=True).stdout.strip()
    mod = types.ModuleType("quick_fix_engine")
    mod.fetch_patch = lambda patch_id: FAIL_PATCH
    monkeypatch.setitem(sys.modules, "quick_fix_engine", mod)
    patch_module = _load_patch_module()
    with pytest.raises(RuntimeError):
        patch_module.apply_patch(1, repo)
    head_after = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True, check=True).stdout.strip()
    assert head_before == head_after
    assert (repo / "file.txt").read_text() == "hello\n"


def test_apply_patch_dirty_worktree(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    (repo / "file.txt").write_text("dirty\n")
    mod = types.ModuleType("quick_fix_engine")
    mod.fetch_patch = lambda patch_id: SUCCESS_PATCH
    monkeypatch.setitem(sys.modules, "quick_fix_engine", mod)
    patch_module = _load_patch_module()
    with pytest.raises(RuntimeError) as exc:
        patch_module.apply_patch(1, repo)
    assert "worktree" in str(exc.value)
