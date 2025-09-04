import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "tools" / "check_static_paths.py"  # path-ignore
FILES = [
    ROOT / "sandbox_runner" / "orphan_discovery.py",  # path-ignore
    ROOT / "self_coding_engine.py",  # path-ignore
    ROOT / "prompt_engine.py",  # path-ignore
    ROOT / "prompt_optimizer.py",  # path-ignore
    ROOT / "prompt_memory_trainer.py",  # path-ignore
]


def test_no_static_py_paths() -> None:
    for file in FILES:
        result = subprocess.run(
            [sys.executable, str(CHECKER), str(file)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert result.stdout.strip() == ""
        assert result.stderr.strip() == ""
