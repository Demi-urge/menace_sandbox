from __future__ import annotations

import sys
import types
import subprocess
import shutil
from pathlib import Path

import self_improvement.target_region as targeting
from tests.self_improvement.test_patch_application import _load_patch_module


BUGGY_SRC = """\
def divide(a, b):
    result = a / 0
    return result
"""

PATCH = """\
diff --git a/buggy.py b/buggy.py  # path-ignore
--- a/buggy.py  # path-ignore
+++ b/buggy.py  # path-ignore
@@ -1,3 +1,3 @@
 def divide(a, b):
-    result = a / 0
+    result = a / b
     return result
"""


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "you@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Your Name"], cwd=repo, check=True)
    (repo / "buggy.py").write_text(BUGGY_SRC)  # path-ignore
    subprocess.run(["git", "add", "buggy.py"], cwd=repo, check=True)  # path-ignore
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)
    return repo


def test_extract_target_region_and_patch(monkeypatch, tmp_path):
    repo = _init_repo(tmp_path)

    trace = subprocess.run(
        [
            sys.executable,
            "-c",
            "import buggy, traceback\n"
            "try:\n buggy.divide(1,0)\n"
            "except Exception:\n print(traceback.format_exc())",
        ],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    ).stdout

    monkeypatch.setattr(targeting, "__file__", str(repo / "__init__.py"))  # path-ignore
    region = targeting.extract_target_region(trace)
    assert region is not None
    assert Path(region.filename) == repo / "buggy.py"  # path-ignore
    assert region.function == "divide"
    assert region.start_line == 1
    assert region.end_line == 3

    before = (repo / "buggy.py").read_text().splitlines()  # path-ignore
    pycache = repo / "__pycache__"
    if pycache.exists():
        shutil.rmtree(pycache)

    mod = types.ModuleType("quick_fix_engine")
    mod.fetch_patch = lambda patch_id: PATCH
    monkeypatch.setitem(sys.modules, "quick_fix_engine", mod)
    patch_module = _load_patch_module()
    commit, diff = patch_module.apply_patch(1, repo)
    assert len(commit) == 40
    assert diff == PATCH

    after = (repo / "buggy.py").read_text().splitlines()  # path-ignore
    assert after[1] == "    result = a / b"
    assert before[0] == after[0]
    assert before[2] == after[2]
