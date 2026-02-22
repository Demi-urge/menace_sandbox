from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_presets.py"
FIXTURES = ROOT / "tests" / "fixtures" / "presets"


def test_clean_preset_passes() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(FIXTURES / "clean_preset.json")],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "preflight passed" in result.stdout.lower()


def test_user_misuse_preset_fails() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(FIXTURES / "user_misuse_preset.json")],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "SCENARIO_NAME" in result.stderr
    assert "FAILURE_MODES" in result.stderr
    assert "user_misuse" in result.stderr
