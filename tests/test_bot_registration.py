import subprocess
import sys
from pathlib import Path

def test_bot_classes_are_managed():
    script = Path(__file__).resolve().parents[1] / "tools" / "check_self_coding_registration.py"
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
