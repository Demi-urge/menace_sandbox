import subprocess
import sys
from pathlib import Path


def test_context_builder_static_analysis():
    script = Path(__file__).resolve().parents[1] / "scripts" / "check_context_builder_usage.py"
    subprocess.run([sys.executable, str(script)], check=True)
