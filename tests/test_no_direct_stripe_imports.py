import subprocess
import sys
from pathlib import Path

from dynamic_path_router import resolve_path


def test_no_direct_stripe_imports() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    files = subprocess.check_output(
        ["git", "ls-files", "*.py"],
        cwd=repo_root,
        text=True,
    ).splitlines()
    script = resolve_path("scripts/check_stripe_imports.py")
    result = subprocess.run(
        [sys.executable, str(script), *files],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
