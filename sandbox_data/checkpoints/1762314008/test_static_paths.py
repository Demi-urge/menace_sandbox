import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "tools" / "check_static_paths.py"  # path-ignore

TARGETS = [ROOT / "sandbox_runner", *ROOT.glob("self_*")]
FILES = sorted(
    {
        f
        for target in TARGETS
        for f in ([target] if target.is_file() else target.rglob("*.py"))  # path-ignore
        if f.suffix == ".py"  # path-ignore
    }
)


def test_no_static_py_paths() -> None:
    result = subprocess.run(
        [sys.executable, str(CHECKER), *map(str, FILES)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == ""
    assert result.stderr.strip() == ""
