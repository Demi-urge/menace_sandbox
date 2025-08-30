"""Lightweight test harness for executing repository tests in isolation.

The harness clones the provided repository into a temporary directory,
creates an ephemeral :mod:`venv` and installs dependencies from
``requirements.txt``.  Unit tests are executed via :mod:`pytest` and the
captured output is returned for further analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import tempfile
import time


@dataclass
class TestHarnessResult:
    """Result information returned by :func:`run_tests`.

    Attributes
    ----------
    success:
        ``True`` when the test suite exited with a ``0`` status code.
    stdout, stderr:
        Captured streams from the test execution.  Dependency installation and
        cloning logs are merged into ``stdout`` for convenience.
    duration:
        Total runtime in seconds.
    """

    success: bool
    stdout: str
    stderr: str
    duration: float

    def __bool__(self) -> bool:  # pragma: no cover - trivial
        return self.success


def _python_bin(venv: Path) -> Path:
    """Return the path to the Python executable inside ``venv``."""

    if sys.platform == "win32":  # pragma: no cover - Windows not used in tests
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def run_tests(repo_path: Path, changed_path: Path | None = None) -> TestHarnessResult:
    """Execute unit tests for ``repo_path`` inside an isolated environment.

    Parameters
    ----------
    repo_path:
        Path to the repository that should be tested.  The repository is
        cloned into a temporary directory to avoid side effects.
    changed_path:
        Path to the patched file.  Currently unused but accepted to mirror the
        interface expected by :meth:`SelfCodingEngine._run_ci`.
    """

    start = time.time()
    tmpdir_obj = tempfile.TemporaryDirectory(prefix="ci-")
    tmpdir = Path(tmpdir_obj.name)
    stdout_parts: list[str] = []
    try:
        clone = subprocess.run(
            ["git", "clone", str(repo_path), str(tmpdir)],
            capture_output=True,
            text=True,
        )
        stdout_parts.append(clone.stdout)
        stdout_parts.append(clone.stderr)

        venv_dir = tmpdir / "venv"
        subprocess.run(
            [sys.executable, "-m", "venv", str(venv_dir)],
            check=True,
        )
        python = _python_bin(venv_dir)

        req_file = tmpdir / "requirements.txt"
        if req_file.exists():
            install = subprocess.run(
                [str(python), "-m", "pip", "install", "-r", "requirements.txt"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
            )
            stdout_parts.append(install.stdout)
            stdout_parts.append(install.stderr)

        tests = subprocess.run(
            [str(python), "-m", "pytest", "-q"],
            cwd=tmpdir,
            capture_output=True,
            text=True,
        )

        duration = time.time() - start
        stdout_parts.append(tests.stdout)
        return TestHarnessResult(
            success=tests.returncode == 0,
            stdout="".join(stdout_parts),
            stderr=tests.stderr,
            duration=duration,
        )
    finally:
        tmpdir_obj.cleanup()

