"""Stub coordinator used for tests.

Paths are resolved relative to the repository root.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from typing import Iterable, List, Optional, Sequence

try:  # pragma: no cover - support package and flat layouts
    from ..dynamic_path_router import resolve_path
except Exception:  # pragma: no cover - fallback when executed directly
    from dynamic_path_router import resolve_path  # type: ignore


def run_script(
    script: str, env: Optional[dict] = None, args: Optional[List[str]] = None
) -> subprocess.CompletedProcess:
    """Run a python script while merging given environment variables."""

    args = args or []
    env_vars = os.environ.copy()
    if env:
        env_vars.update(env)

    if not os.path.exists(script):
        raise FileNotFoundError(script)

    return subprocess.run(["python", script, *args], check=True, env=env_vars)


def run_scripts(
    scripts: Sequence[str],
    *,
    env: Optional[dict] = None,
    parallel: bool = False,
) -> List[subprocess.CompletedProcess]:
    """Run multiple scripts either sequentially or in parallel."""

    processes: List[subprocess.CompletedProcess] = []
    if parallel:
        procs: List[subprocess.Popen] = []
        for script in scripts:
            env_vars = os.environ.copy()
            if env:
                env_vars.update(env)
            procs.append(
                subprocess.Popen(["python", script], env=env_vars)
            )
        for proc in procs:
            proc.wait()
            processes.append(
                subprocess.CompletedProcess(proc.args, proc.returncode)
            )
    else:
        for script in scripts:
            processes.append(run_script(script, env=env))
    return processes


def main(argv: Optional[Iterable[str]] = None) -> None:
    """Entry point executing one or more scripts."""

    parser = argparse.ArgumentParser(description="Run Menace scripts")
    parser.add_argument(
        "scripts",
        nargs="*",
        default=[str(resolve_path("menace_master.py"))],
        help=(
            "Python scripts to run. Paths are resolved relative to the "
            "repository root (default: menace_master.py)."
        ),
    )
    parser.add_argument("--env", action="append", default=[], help="KEY=VALUE")
    parser.add_argument("--parallel", action="store_true", help="Run scripts in parallel")
    args = parser.parse_args(list(argv) if argv is not None else None)
    scripts = [str(resolve_path(s)) for s in args.scripts]

    env = dict(item.split("=", 1) for item in args.env)
    if args.parallel:
        run_scripts(scripts, env=env, parallel=True)
    else:
        run_scripts(scripts, env=env)


__all__ = ["run_script", "run_scripts", "main"]
