import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "tools" / "check_dynamic_paths.py"  # path-ignore


@pytest.mark.parametrize("filename", [
    "config.yaml",
    "data.json",
    "records.db",
])
def test_extension_triggers(tmp_path, filename) -> None:
    target = tmp_path / "sample.py"  # path-ignore
    target.write_text(f'PATH = "{filename}"\n', encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(CHECKER), str(target)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "missing resolve_path" in result.stdout


def test_sandbox_settings_yaml_trigger(tmp_path) -> None:
    target = tmp_path / "sample.py"  # path-ignore
    target.write_text('NAME = "sandbox_settings.yaml"\n', encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(CHECKER), str(target)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "missing resolve_path" in result.stdout


def test_resolve_path_suppresses_warning(tmp_path) -> None:
    target = tmp_path / "sample.py"  # path-ignore
    target.write_text(
        'NAME = "sandbox_settings.yaml"\nresolve_path("sandbox_settings.yaml")\n',
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(CHECKER), str(target)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
