import subprocess
import sys
from pathlib import Path


def test_no_unmanaged_bots():
    """Fail if any *_bot.py modules lack @self_coding_managed."""
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "tools" / "find_unmanaged_bots.py"
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
