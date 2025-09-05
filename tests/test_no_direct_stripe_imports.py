import subprocess
import sys
from pathlib import Path

from dynamic_path_router import resolve_path


def test_no_direct_stripe_imports() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    files = subprocess.check_output(
        ["git", "ls-files", "*.py"],  # path-ignore
        cwd=repo_root,
        text=True,
    ).splitlines()
    script = resolve_path("scripts/check_stripe_imports.py")  # path-ignore
    result = subprocess.run(
        [sys.executable, str(script), *files],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_payment_keywords_require_router(tmp_path) -> None:
    mod = tmp_path / "payment_mod.py"  # path-ignore
    mod.write_text("def charge_user(x):\n    return x\n")
    script = resolve_path("scripts/check_stripe_imports.py")  # path-ignore
    result = subprocess.run(
        [sys.executable, str(script), str(mod)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
