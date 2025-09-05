import subprocess
import sys
from pathlib import Path

from dynamic_path_router import resolve_path


def test_no_raw_stripe_usage() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = resolve_path("scripts/check_raw_stripe_usage.py")  # path-ignore
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
