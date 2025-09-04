#!/usr/bin/env python3
"""Simple benchmark for sandbox_runner.orphan_discovery.

Run as::

    python scripts/orphan_discovery_benchmark.py <repo_path>

The script executes :func:`discover_recursive_orphans` twice: once with a
single worker and once with the number of workers defined by
``SANDBOX_DISCOVERY_WORKERS`` or the CPU count.  Durations for both runs are
printed to stdout.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import importlib.util
import types

from dynamic_path_router import resolve_path


REPO_ROOT = resolve_path(".")

pkg = types.ModuleType("sandbox_runner")
pkg.__path__ = [str(resolve_path("sandbox_runner"))]
sys.modules["sandbox_runner"] = pkg

mod_spec = importlib.util.spec_from_file_location(
    "sandbox_runner.orphan_discovery",
    resolve_path("sandbox_runner/orphan_discovery.py"),
)
orphan_discovery = importlib.util.module_from_spec(mod_spec)
assert mod_spec.loader
mod_spec.loader.exec_module(orphan_discovery)
sys.modules["sandbox_runner.orphan_discovery"] = orphan_discovery
discover_recursive_orphans = orphan_discovery.discover_recursive_orphans


def _time_run(repo: str) -> float:
    start = time.perf_counter()
    discover_recursive_orphans(repo)
    return time.perf_counter() - start


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("Usage: orphan_discovery_benchmark.py <repo_path>")
        return 1

    repo = str(Path(argv[0]).resolve())

    os.environ["SANDBOX_DISCOVERY_WORKERS"] = "1"
    sequential = _time_run(repo)

    workers = int(os.getenv("SANDBOX_DISCOVERY_WORKERS", "0"))
    if workers <= 1:
        os.environ["SANDBOX_DISCOVERY_WORKERS"] = str(os.cpu_count() or 1)
    parallel = _time_run(repo)

    print(f"Sequential (1 worker): {sequential:.2f}s")
    print(f"Parallel ({os.environ['SANDBOX_DISCOVERY_WORKERS']} workers): {parallel:.2f}s")
    return 0


if __name__ == "__main__":  # pragma: no cover - manual usage
    raise SystemExit(main())
