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
import json
import urllib.parse
from typing import Any
from contextlib import contextmanager

from dynamic_path_router import resolve_path, path_for_prompt


from ..error_parser import ErrorParser
from sandbox_settings import SandboxSettings
try:  # pragma: no cover - tolerate trimmed environments
    from . import environment as _environment
except Exception as exc:  # pragma: no cover - optional dependency fallback
    _environment = None
    _ENV_IMPORT_ERROR = exc
else:  # pragma: no branch - bookkeeping
    _ENV_IMPORT_ERROR = None

if _environment is not None:
    get_edge_case_stubs = getattr(_environment, "get_edge_case_stubs", lambda: {})
    preserve_sandbox_env = _environment.preserve_sandbox_env
    cleanup_artifacts = _environment.cleanup_artifacts
else:

    def get_edge_case_stubs() -> dict[str, Any]:  # pragma: no cover - simple fallback
        """Return an empty edge-case stub map when environment helpers are missing."""

        return {}


    @contextmanager
    def preserve_sandbox_env(*_args, **_kwargs):  # pragma: no cover - fallback
        """Raise a helpful error when sandbox environment helpers are unavailable."""

        if _ENV_IMPORT_ERROR is not None:
            raise RuntimeError(
                "sandbox_runner.environment helpers are unavailable; install optional "
                "sandbox dependencies"
            ) from _ENV_IMPORT_ERROR
        yield


    def cleanup_artifacts(*_args, **_kwargs):  # pragma: no cover - noop fallback
        """No-op cleanup when environment helpers are missing."""

        return None

from .scoring import record_run


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
    entropy_delta:
        Change in entropy observed during the test run when coverage
        information is available. ``None`` when entropy metrics could not be
        determined.
    """

    success: bool
    stdout: str
    stderr: str
    duration: float
    failure: dict | None = None
    path: str | None = None
    stub: dict | None = None
    preset: dict | None = None
    coverage: dict | None = None
    edge_cases: dict | None = None
    entropy_delta: float | None = None
    executed_functions: list[str] | None = None

    def __bool__(self) -> bool:  # pragma: no cover - trivial
        return self.success


def _python_bin(venv: Path) -> Path:
    """Return the path to the Python executable inside ``venv``."""
    subdir = "Scripts" if sys.platform == "win32" else "bin"
    exe = "python.exe" if sys.platform == "win32" else "python"
    return resolve_path(venv / subdir / exe)


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


def _run_once(
    repo_path: Path,
    changed_path: Path | None = None,
    *,
    backend: str = "venv",
    stub: dict | None = None,
    preset: dict | None = None,
    edge_cases: dict | None = None,
    clone_repo: bool = True,
    write_edge_cases: bool = True,
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

    repo_path = Path(resolve_path(str(repo_path)))
    if changed_path:
        try:
            changed_path = Path(resolve_path(str(changed_path)))
        except FileNotFoundError:
            changed_path = Path(changed_path)

    start = time.time()
    if clone_repo:
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
        except Exception:
            tmpdir_obj.cleanup()
            raise
    else:
        tmpdir_obj = None
        tmpdir = repo_path
        stdout_parts = []

    try:
        with preserve_sandbox_env():
            if backend not in {"venv", "docker"}:
                raise ValueError(f"unknown backend: {backend}")

            edge_data = edge_cases
            if write_edge_cases:
                if edge_data is None:
                    raw = os.getenv("SANDBOX_EDGE_CASES")
                    if raw:
                        try:
                            parsed = json.loads(raw)
                            if isinstance(parsed, dict):
                                edge_data = parsed
                        except Exception:
                            edge_data = None
                if edge_data:
                    for name, payload in edge_data.items():
                        try:
                            scheme = urllib.parse.urlparse(name).scheme
                            if scheme in {"http", "https"}:
                                continue
                            dest = tmpdir / name
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            content = (
                                payload if isinstance(payload, str) else json.dumps(payload)
                            )
                            dest.write_text(content, encoding="utf-8")
                        except Exception:
                            pass

            if edge_data is not None:
                os.environ["SANDBOX_EDGE_CASES"] = json.dumps(edge_data)
            else:
                os.environ.pop("SANDBOX_EDGE_CASES", None)

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
            cov_env = os.getenv("SANDBOX_CAPTURE_COVERAGE")
            capture_cov = True
            if cov_env is not None and cov_env.lower() in {"0", "false", "no"}:
                capture_cov = False
            if backend == "venv":
                venv_dir = tmpdir / "venv"
                subprocess.run(
                    [sys.executable, "-m", "venv", str(venv_dir)],
                    check=True,
                )
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

                if capture_cov:
                    check_cov = subprocess.run(
                        [str(python), "-c", "import coverage"],
                        capture_output=True,
                        text=True,
                    )
                    if check_cov.returncode != 0:
                        cov_install = subprocess.run(
                            [str(python), "-m", "pip", "install", "coverage"],
                            cwd=tmpdir,
                            capture_output=True,
                            text=True,
                        )
                        stdout_parts.append(cov_install.stdout)
                        stdout_parts.append(cov_install.stderr)
                        if cov_install.returncode != 0:
                            capture_cov = False

                pytest_args = ["-q", "--tb=short", "-p", "sandbox_runner.edge_case_plugin"]
                if rel_paths:
                    try:
                        rel_paths = [p.relative_to(repo_path) for p in rel_paths]
                    except Exception as exc:
                        logger.warning(
                            "Failed to resolve relative paths %s: %s",
                            [path_for_prompt(p) for p in rel_paths],
                            exc,
                        )
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
                    cov_json = tmpdir / "cov.json"
                    snippet = (
                        "import sys,coverage,pytest;"
                        "cov=coverage.Coverage();cov.start();"
                        "rc=pytest.main(sys.argv[1:]);"
                        "cov.stop();cov.json_report(outfile='cov.json');"
                        "sys.exit(rc)"
                    )
                    pytest_cmd = [str(python), "-c", snippet, *pytest_args]
                else:
                    pytest_cmd = [str(python), "-m", "pytest", *pytest_args]

                tests = subprocess.run(
                    pytest_cmd,
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                )
            else:  # backend == "docker"
                pytest_args = [
                    "-m",
                    "pytest",
                    "-q",
                    "--tb=short",
                    "-p",
                    "sandbox_runner.edge_case_plugin",
                ]
                if rel_paths:
                    try:
                        rel_paths = [p.relative_to(repo_path) for p in rel_paths]
                    except Exception as exc:
                        logger.warning(
                            "Failed to resolve relative paths %s: %s",
                            [path_for_prompt(p) for p in rel_paths],
                            exc,
                        )
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
                    inner_cmds.append("pip install coverage")
                    snippet = (
                        "python - <<'PY'\n"
                        "import sys,coverage,pytest\n"
                        "cov=coverage.Coverage()\n"
                        "cov.start()\n"
                        "rc=pytest.main(sys.argv[1:])\n"
                        "cov.stop()\n"
                        "cov.json_report(outfile='cov.json')\n"
                        "sys.exit(rc)\n"
                        "PY " + " ".join(pytest_args)
                    )
                    inner_cmds.append(snippet)
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
                ]
                for key in (
                    "SANDBOX_INPUT_STUBS",
                    "SANDBOX_ENV_PRESETS",
                    "SANDBOX_EDGE_CASES",
                ):
                    val = os.environ.get(key)
                    if val is not None:
                        docker_cmd.extend(["-e", f"{key}={val}"])
                docker_cmd.extend(
                    [
                        "python:3.11-slim",
                        "bash",
                        "-lc",
                        " && ".join(inner_cmds),
                    ]
                )
                tests = subprocess.run(
                    docker_cmd,
                    capture_output=True,
                    text=True,
                )

            duration = time.time() - start
            stdout_parts.append(tests.stdout)
            failure = None
            cov_json = tmpdir / "cov.json"
            cov_data = None
            executed_functions: list[str] | None = None
            coverage_map: dict[str, list[str]] | None = None
            if cov_json.exists():
                try:
                    cov_data = json.loads(cov_json.read_text())
                except Exception:
                    cov_data = None
                if cov_data is not None:
                    try:
                        from .environment import load_coverage_report  # type: ignore

                        executed_functions = load_coverage_report(cov_data)
                        coverage_map = {}
                        for func in executed_functions:
                            try:
                                path, fn = func.split(":", 1)
                            except ValueError:
                                continue
                            coverage_map.setdefault(path, []).append(fn)
                    except Exception:
                        executed_functions = []
                        coverage_map = {}
                    cov_data["executed_functions"] = executed_functions
            entropy_delta = None
            if cov_data is not None:
                try:
                    from self_improvement.metrics import (
                        compute_entropy_metrics,
                        compute_entropy_delta,
                    )

                    files: list[Path] = []
                    for f in cov_data.get("files", {}):
                        try:
                            rel = Path(f).relative_to(tmpdir)
                            files.append(repo_path / rel)
                        except Exception:
                            continue
                    if files:
                        code_diversity, token_complexity, _ = compute_entropy_metrics(files)
                        entropy_delta, _ = compute_entropy_delta(
                            code_diversity, token_complexity
                        )
                except Exception:
                    pass
            if tests.returncode != 0:
                failure = ErrorParser.parse(tests.stdout + tests.stderr)
                frames = _extract_frames(failure.get("trace", ""))
                if frames:
                    failure["frames"] = frames
                    last = frames[-1]
                    failure["file"] = last["file"]
                    failure["line"] = last["line"]
                    failure["function"] = last["function"]
            res = TestHarnessResult(
                success=tests.returncode == 0,
                stdout="".join(stdout_parts),
                stderr=tests.stderr,
                duration=duration,
                failure=failure,
                path=selected,
                stub=stub,
                preset=preset,
                coverage=cov_data,
                edge_cases=edge_data,
                entropy_delta=entropy_delta,
                executed_functions=executed_functions,
            )
            record_run(
                result=res,
                metrics={
                    "runtime": duration,
                    "entropy_delta": entropy_delta,
                    "coverage": coverage_map,
                    "executed_functions": executed_functions,
                    "failure": failure,
                },
            )
            return res
    finally:
        if tmpdir_obj is not None:
            try:
                tmpdir_obj.cleanup()
            except Exception:
                logger.warning("tempdir cleanup failed", exc_info=True)
            cleanup_artifacts([tmpdir])
        else:
            cleanup_artifacts()


def run_tests(
    repo_path: Path,
    changed_path: Path | None = None,
    *,
    backend: str = "venv",
    input_stubs: list[dict[str, Any]] | None = None,
    presets: list[dict[str, Any]] | None = None,
) -> TestHarnessResult | list[TestHarnessResult]:
    """Execute tests for ``repo_path`` across ``input_stubs`` and ``presets``.

    When ``input_stubs`` is not supplied the harness generates hostile and
    misuse payloads via :func:`sandbox_runner.environment.generate_input_stubs`.
    Each stub/preset combination executes in a fresh repository clone with the
    dictionaries serialised into ``SANDBOX_INPUT_STUBS`` and
    ``SANDBOX_ENV_PRESETS`` environment variables.  Per-run results are recorded
    via the unified scoring module.
    """

    if input_stubs is None:
        try:
            from .environment import generate_input_stubs

            input_stubs = generate_input_stubs(1, strategy="hostile")
            input_stubs += generate_input_stubs(1, strategy="misuse")
        except Exception:
            input_stubs = [{}]
    if not input_stubs:
        input_stubs = [{}]

    if not presets:
        presets = [{}]
    settings = SandboxSettings()
    inject_edges = getattr(settings, "inject_edge_cases", True)

    results: list[TestHarnessResult] = []
    for stub in input_stubs:
        for preset in presets:
            edge_cases: dict[str, Any] = {}
            if inject_edges:
                try:
                    edge_cases = get_edge_case_stubs()
                except Exception:
                    edge_cases = {}
            with tempfile.TemporaryDirectory(prefix="repo-") as repodir:
                repo_tmp = Path(repodir)
                clone = subprocess.run(
                    ["git", "clone", str(repo_path), str(repo_tmp)],
                    capture_output=True,
                    text=True,
                )
                if clone.returncode != 0:
                    msg = f"git clone failed with code {clone.returncode}: {clone.stderr}"
                    logger.error(msg.strip())
                    raise RuntimeError(msg)
                if edge_cases:
                    for name, payload in edge_cases.items():
                        try:
                            scheme = urllib.parse.urlparse(name).scheme
                            if scheme in {"http", "https"}:
                                continue
                            dest = repo_tmp / name
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            content = (
                                payload if isinstance(payload, str) else json.dumps(payload)
                            )
                            dest.write_text(content, encoding="utf-8")
                        except Exception:
                            pass

                tmp_changed: Path | None = None
                if changed_path:
                    try:
                        rel = changed_path.relative_to(repo_path)
                        tmp_changed = repo_tmp / rel
                        if changed_path.is_file() and changed_path.suffix == ".txt":
                            tmp_changed.write_text(
                                changed_path.read_text(encoding="utf-8"),
                                encoding="utf-8",
                            )
                    except Exception:
                        tmp_changed = changed_path

                with preserve_sandbox_env():
                    os.environ["SANDBOX_INPUT_STUBS"] = json.dumps([stub])
                    os.environ["SANDBOX_ENV_PRESETS"] = json.dumps([preset])
                    res = _run_once(
                        repo_tmp,
                        tmp_changed,
                        backend=backend,
                        stub=stub,
                        preset=preset,
                        edge_cases=edge_cases,
                        clone_repo=False,
                        write_edge_cases=False,
                    )
                record_run(
                    result=res,
                    metrics={
                        "coverage": res.coverage,
                        "executed_functions": res.executed_functions,
                        "entropy_delta": res.entropy_delta,
                        "runtime": res.duration,
                        "failure": res.failure,
                    },
                )
                results.append(res)
            cleanup_artifacts([repo_tmp])
    if len(results) == 1:
        return results[0]
    return results
