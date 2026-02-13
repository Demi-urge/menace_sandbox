from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest


def _find_python311() -> str | None:
    candidates = [
        shutil.which("python3.11"),
        str(Path.home() / ".pyenv" / "versions" / "3.11.14" / "bin" / "python"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if not path.exists():
            continue
        check = subprocess.run(
            [str(path), "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if check.returncode == 0 and "3.11" in check.stdout:
            return str(path)
    return None


def test_built_wheel_contains_packaged_self_debugger_module(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    builder_python = _find_python311()
    if builder_python is None:
        pytest.skip("python3.11 is required to build the project wheel")

    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()

    subprocess.run(
        [
            builder_python,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(dist_dir),
        ],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    wheels = sorted(dist_dir.glob("menace-*.whl"))
    assert wheels, "expected pip wheel build to produce a menace wheel"

    with zipfile.ZipFile(wheels[0]) as built_wheel:
        wheel_entries = set(built_wheel.namelist())

    assert "menace/self_debugger_sandbox.py" in wheel_entries
    assert "sandbox_runner/import_candidates.py" in wheel_entries
