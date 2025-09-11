import subprocess
import sys
from pathlib import Path


def test_self_coding_registration_script():
    """Ensure all *Bot classes are properly registered or decorated."""
    script = Path(__file__).resolve().parents[1] / "tools" / "check_self_coding_registration.py"
    subprocess.run([sys.executable, str(script)], check=True)
