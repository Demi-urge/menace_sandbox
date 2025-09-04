"""Lightweight test harness for executing repository tests in isolation.

The harness clones the provided repository into a temporary directory,
creates an ephemeral :mod:`venv` and installs dependencies from
``requirements.txt``.  Unit tests are executed via :mod:`pytest` and the
captured output is returned for further analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import re
import os
import subprocess
import sys
import tempfile
import time

from ..error_parser import ErrorParser


logger = logging.getLogger(__name__)


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
    failure: dict | None = None
    path: str | None = None

    def __bool__(self) -> bool:  # pragma: no cover - trivial
        return self.success


def _python_bin(venv: Path) -> Path:
    """Return the path to the Python executable inside ``venv``."""

    if sys.platform == "win32":  # pragma: no cover - Windows not used in tests
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


_FRAME_RE = re.compile(r'File "([^"]+)", line (\d+), in (.+)')
_FRAME_RE_ALT = re.compile(r'([^:\s]+\.py):(\d+): in (.+)')


def _extract_frames(trace: str) -> list[dict[str, str]]:
    """Parse ``trace`` and return structured frame information."""

    frames: list[dict[str, str]] = []
    for line in trace.splitlines():
        line = line.strip()
        m = _FRAME_RE.match(line)
        if m:
            frames.append({"file": m.group(1), "line": m.group(2), "function": m.group(3)})
            continue
        m = _FRAME_RE_ALT.match(line)
        if m:
            frames.append({"file": m.group(1), "line": m.group(2), "function": m.group(3)})
    return frames


def run_tests(
    repo_path: Path,
    changed_path: Path | None = None,
    *,
    backend: str = "venv",
) -> TestHarnessResult:
    """Execute unit tests for ``repo_path`` inside an isolated environment.

    Parameters
    ----------
    repo_path:
        Path to the repository that should be tested.  The repository is
        cloned into a temporary directory to avoid side effects.
    changed_path:
        Optional path to the patched file.  When supplied the pytest invocation
        is narrowed to the impacted tests by either executing the specific test
        file or applying a ``-k`` expression based on the file stem.  When
        multiple paths are supplied via a newline separated file, a combined
        ``-k`` expression is used.
    backend:
        Execution backend.  ``"venv"`` creates a virtual environment and runs
        tests within it.  ``"docker"`` executes tests inside a temporary Docker
        container.  Defaults to ``"venv"`` for backward compatibility.
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
        if clone.returncode != 0:
            msg = f"git clone failed with code {clone.returncode}: {clone.stderr}"
            logger.error(msg.strip())
            raise RuntimeError(msg)

        if backend not in {"venv", "docker"}:
            raise ValueError(f"unknown backend: {backend}")

        req_file = tmpdir / "requirements.txt"
        rel_paths: list[Path] = []
        if changed_path:
            # ``changed_path`` may be either a single file or a file containing
            # a newline separated list of changed files.  The latter is used
            # when multiple files are touched by a patch.
            try:
                if changed_path.is_file() and changed_path.suffix == ".txt":
                    lines = [
                        Path(line.strip())
                        for line in changed_path.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    ]
                    rel_paths.extend(lines)
                else:
                    rel_paths.append(changed_path)
            except Exception:
                rel_paths.append(changed_path)

        selected: str | None = None
        capture_cov = os.getenv("SANDBOX_CAPTURE_COVERAGE")
        if backend == "venv":
            venv_dir = tmpdir / "venv"
            subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
            python = _python_bin(venv_dir)

            if req_file.exists():
                install = subprocess.run(
                    [str(python), "-m", "pip", "install", "-r", "requirements.txt"],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                )
                stdout_parts.append(install.stdout)
                stdout_parts.append(install.stderr)
                if install.returncode != 0:
                    msg = (
                        "dependency installation failed with code "
                        f"{install.returncode}: {install.stderr}"
                    )
                    logger.error(msg.strip())
                    raise RuntimeError(msg)

            pytest_args = ["-q", "--tb=short"]
            if rel_paths:
                try:
                    rel_paths = [p.relative_to(repo_path) for p in rel_paths]
                except Exception as exc:
                    logger.warning("Failed to resolve relative paths %s: %s", rel_paths, exc)
                if len(rel_paths) == 1:
                    rel = rel_paths[0]
                    selected = rel.as_posix()
                    if (
                        "tests" in rel.parts
                        or rel.name.startswith("test_")
                    ) and rel.suffix == ".py":
                        pytest_args.insert(0, selected)
                    else:
                        pytest_args.extend(["-k", rel.stem])
                else:
                    expr = " or ".join(p.stem for p in rel_paths)
                    pytest_args.extend(["-k", expr])
                    selected = " ".join(p.as_posix() for p in rel_paths)

            if capture_cov:
                cov_file = tmpdir / ".coverage"
                pytest_cmd = [
                    str(python),
                    "-m",
                    "coverage",
                    "run",
                    f"--data-file={cov_file}",
                    "-m",
                    "pytest",
                    *pytest_args,
                ]
            else:
                pytest_cmd = [str(python), "-m", "pytest", *pytest_args]

            tests = subprocess.run(
                pytest_cmd,
                cwd=tmpdir,
                capture_output=True,
                text=True,
            )
        else:  # backend == "docker"
            pytest_args = ["-m", "pytest", "-q", "--tb=short"]
            if rel_paths:
                try:
                    rel_paths = [p.relative_to(repo_path) for p in rel_paths]
                except Exception as exc:
                    logger.warning("Failed to resolve relative paths %s: %s", rel_paths, exc)
                if len(rel_paths) == 1:
                    rel = rel_paths[0]
                    selected = rel.as_posix()
                    if (
                        "tests" in rel.parts
                        or rel.name.startswith("test_")
                    ) and rel.suffix == ".py":
                        pytest_args.insert(1, selected)
                    else:
                        pytest_args.extend(["-k", rel.stem])
                else:
                    expr = " or ".join(p.stem for p in rel_paths)
                    pytest_args.extend(["-k", expr])
                    selected = " ".join(p.as_posix() for p in rel_paths)

            inner_cmds: list[str] = []
            if req_file.exists():
                inner_cmds.append("pip install -r requirements.txt")
            if capture_cov:
                inner_cmds.append("coverage run --data-file=.coverage " + " ".join(pytest_args))
            else:
                inner_cmds.append("python " + " ".join(pytest_args))
            docker_cmd = [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{tmpdir}:/repo",
                "-w",
                "/repo",
                "python:3.11-slim",
                "bash",
                "-lc",
                " && ".join(inner_cmds),
            ]
            tests = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
            )

        duration = time.time() - start
        stdout_parts.append(tests.stdout)
        failure = None
        if tests.returncode != 0:
            failure = ErrorParser.parse(tests.stdout + tests.stderr)
            frames = _extract_frames(failure.get("trace", ""))
            if frames:
                failure["frames"] = frames
                last = frames[-1]
                failure["file"] = last["file"]
                failure["line"] = last["line"]
                failure["function"] = last["function"]
        return TestHarnessResult(
            success=tests.returncode == 0,
            stdout="".join(stdout_parts),
            stderr=tests.stderr,
            duration=duration,
            failure=failure,
            path=selected,
        )
    finally:
        tmpdir_obj.cleanup()
