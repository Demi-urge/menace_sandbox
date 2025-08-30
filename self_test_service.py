# flake8: noqa
from __future__ import annotations

"""Service running self tests on a schedule."""

import asyncio
import time
from asyncio import Lock
from filelock import FileLock
import json
import logging
import os
import shlex
import sqlite3
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
import threading
import inspect
import subprocess
import ast

from db_router import init_db_router

MENACE_ID = uuid.uuid4().hex
LOCAL_DB_PATH = os.getenv("MENACE_LOCAL_DB_PATH", f"./menace_{MENACE_ID}_local.db")
SHARED_DB_PATH = os.getenv("MENACE_SHARED_DB_PATH", "./shared/global.db")
GLOBAL_ROUTER = init_db_router(MENACE_ID, LOCAL_DB_PATH, SHARED_DB_PATH)

from orphan_analyzer import classify_module
from sandbox_settings import SandboxSettings
from logging_utils import log_record

if os.getenv("SANDBOX_CENTRAL_LOGGING") == "1":
    from logging_utils import setup_logging

    setup_logging()

try:
    from .data_bot import DataBot
    from .error_bot import ErrorDB
    from .error_logger import ErrorLogger
    from .knowledge_graph import KnowledgeGraph
except Exception:  # pragma: no cover - fallback when imported directly
    from data_bot import DataBot  # type: ignore
    from error_bot import ErrorDB  # type: ignore
    from error_logger import ErrorLogger  # type: ignore
    from knowledge_graph import KnowledgeGraph  # type: ignore

try:
    from .auto_env_setup import get_recursive_isolated, set_recursive_isolated
except Exception:  # pragma: no cover - direct execution
    from auto_env_setup import (  # type: ignore
        get_recursive_isolated,
        set_recursive_isolated,
    )

try:
    from . import metrics_exporter as _me
except Exception:  # pragma: no cover - package may not be available
    import metrics_exporter as _me  # type: ignore

orphan_modules_reintroduced_total = _me.orphan_modules_reintroduced_total
orphan_modules_passed_total = _me.orphan_modules_passed_total
orphan_modules_tested_total = _me.orphan_modules_tested_total
orphan_modules_failed_total = _me.orphan_modules_failed_total
orphan_modules_reclassified_total = _me.orphan_modules_reclassified_total
orphan_modules_redundant_total = _me.orphan_modules_redundant_total
orphan_modules_legacy_total = _me.orphan_modules_legacy_total

router = GLOBAL_ROUTER

settings = SandboxSettings()

self_test_passed_total = _me.Gauge(
    "self_test_passed_total", "Total number of passed self tests"
)
self_test_failed_total = _me.Gauge(
    "self_test_failed_total", "Total number of failed self tests"
)
self_test_average_runtime_seconds = _me.Gauge(
    "self_test_average_runtime_seconds", "Average runtime of the last self test run"
)
self_test_average_coverage = _me.Gauge(
    "self_test_average_coverage", "Average coverage percentage of the last self test run"
)

# Track container-related issues
self_test_container_failures_total = _me.Gauge(
    "self_test_container_failures_total",
    "Total container cleanup/listing failures during self tests",
)
self_test_container_timeouts_total = _me.Gauge(
    "self_test_container_timeouts_total",
    "Total container execution timeouts during self tests",
)

setattr(_me, "self_test_container_failures_total", self_test_container_failures_total)
setattr(_me, "self_test_container_timeouts_total", self_test_container_timeouts_total)

_container_lock = Lock()
_file_lock = FileLock(settings.self_test_lock_file)

# ---------------------------------------------------------------------------
# Track failing critical test suites across runs
CRITICAL_SUITES = {"security", "alignment"}
_FAILED_CRITICAL_TESTS: set[str] = set()


def set_failed_critical_tests(failed: Iterable[str]) -> None:
    """Record names of critical test suites that failed in the last run."""

    global _FAILED_CRITICAL_TESTS
    _FAILED_CRITICAL_TESTS = {
        str(f).lower() for f in failed if str(f).lower() in CRITICAL_SUITES
    }


def get_failed_critical_tests() -> set[str]:
    """Return the set of failing critical test suites."""

    return set(_FAILED_CRITICAL_TESTS)

try:
    from sandbox_runner.dependency_utils import collect_local_dependencies
except Exception:  # pragma: no cover - fallback when sandbox_runner is stubbed
    def collect_local_dependencies(
        paths: Iterable[str],
        *,
        initial_parents: Mapping[str, list[str]] | None = None,
        on_module: Callable[[str, Path, list[str]], None] | None = None,
        on_dependency: Callable[[str, str, list[str]], None] | None = None,
        max_depth: int | None = None,
    ) -> set[str]:
        """Walk imports for ``paths`` to discover local dependencies.

        This simplified implementation mirrors the behaviour of the sandbox
        runner's dependency collector.  It parses ``import`` statements for the
        given modules, recursively following imports that resolve to files within
        the current repository.  ``on_module`` and ``on_dependency`` callbacks
        are invoked on a best-effort basis.

        Raises
        ------
        RuntimeError
            If a provided path does not exist so dependencies cannot be
            determined.
        """

        repo = Path(os.getenv("SANDBOX_REPO_PATH", ".")).resolve()

        queue: list[tuple[Path, list[str]]] = []
        for m in paths:
            p = Path(m)
            if not p.is_absolute():
                p = repo / p
            if not p.exists():
                raise RuntimeError(f"module not found: {m}")
            try:
                rel = p.resolve().relative_to(repo).as_posix()
            except Exception:
                rel = p.as_posix()
            parents = list(initial_parents.get(rel, []) if initial_parents else [])
            queue.append((p, parents))

        seen: set[str] = set()
        while queue:
            path, parents = queue.pop()
            try:
                rel = path.resolve().relative_to(repo).as_posix()
            except Exception:
                rel = path.as_posix()

            if on_module is not None:
                try:
                    on_module(rel, path, parents)
                except Exception:  # pragma: no cover - best effort
                    pass
            if rel in seen:
                continue
            seen.add(rel)

            if max_depth is not None and len(parents) >= max_depth:
                continue

            try:
                src = path.read_text(encoding="utf-8")
                tree = ast.parse(src)
            except Exception:
                continue

            pkg_parts = rel.split("/")[:-1]
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        parts = alias.name.split(".")
                        for cand in (
                            repo / Path(*parts).with_suffix(".py"),
                            repo / Path(*parts) / "__init__.py",
                        ):
                            if cand.exists():
                                dep_rel = cand.resolve().relative_to(repo).as_posix()
                                dep_parents = [rel] + parents
                                if on_dependency is not None:
                                    try:
                                        on_dependency(dep_rel, rel, dep_parents)
                                    except Exception:  # pragma: no cover - best effort
                                        pass
                                queue.append((cand, dep_parents))
                                break
                elif isinstance(node, ast.ImportFrom):
                    if node.level:
                        base_prefix = pkg_parts[: len(pkg_parts) - node.level + 1]
                    else:
                        base_prefix = pkg_parts
                    parts = base_prefix + (node.module.split(".") if node.module else [])
                    candidates = [
                        repo / Path(*parts).with_suffix(".py"),
                        repo / Path(*parts) / "__init__.py",
                    ]
                    dep = next((c for c in candidates if c.exists()), None)
                    if dep is not None:
                        dep_rel = dep.resolve().relative_to(repo).as_posix()
                        dep_parents = [rel] + parents
                        if on_dependency is not None:
                            try:
                                on_dependency(dep_rel, rel, dep_parents)
                            except Exception:  # pragma: no cover - best effort
                                pass
                        queue.append((dep, dep_parents))
                    for alias in node.names:
                        if alias.name == "*":
                            continue
                        sub_parts = parts + alias.name.split(".")
                        for cand in (
                            repo / Path(*sub_parts).with_suffix(".py"),
                            repo / Path(*sub_parts) / "__init__.py",
                        ):
                            if cand.exists():
                                dep_rel = cand.resolve().relative_to(repo).as_posix()
                                dep_parents = [rel] + parents
                                if on_dependency is not None:
                                    try:
                                        on_dependency(dep_rel, rel, dep_parents)
                                    except Exception:  # pragma: no cover - best effort
                                        pass
                                queue.append((cand, dep_parents))
                                break
                elif isinstance(node, ast.Call):
                    mod_name: str | None = None
                    if (
                        isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "importlib"
                        and node.func.attr == "import_module"
                        and node.args
                    ):
                        arg = node.args[0]
                        if isinstance(arg, ast.Str):
                            mod_name = arg.s
                        elif isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            mod_name = arg.value
                    elif (
                        isinstance(node.func, ast.Name)
                        and node.func.id in {"import_module", "__import__"}
                        and node.args
                    ):
                        arg = node.args[0]
                        if isinstance(arg, ast.Str):
                            mod_name = arg.s
                        elif isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            mod_name = arg.value
                    if mod_name:
                        parts = mod_name.split(".")
                        for cand in (
                            repo / Path(*parts).with_suffix(".py"),
                            repo / Path(*parts) / "__init__.py",
                        ):
                            if cand.exists():
                                dep_rel = cand.resolve().relative_to(repo).as_posix()
                                dep_parents = [rel] + parents
                                if on_dependency is not None:
                                    try:
                                        on_dependency(dep_rel, rel, dep_parents)
                                    except Exception:  # pragma: no cover - best effort
                                        pass
                                queue.append((cand, dep_parents))
                                break

        return seen

try:
    from sandbox_runner.orphan_discovery import (
        append_orphan_cache,
        append_orphan_classifications,
        prune_orphan_cache,
        load_orphan_cache,
    )
except Exception:  # pragma: no cover - fallback when sandbox_runner is stubbed
    from orphan_discovery import (
        append_orphan_cache,
        append_orphan_classifications,
        prune_orphan_cache,
        load_orphan_cache,
    )


class SelfTestService:
    """Periodically execute the test suite to validate core bots.

    If ``result_callback`` is provided, it will be invoked with a dictionary
    containing cumulative results each time a test file finishes running and
    again once the entire run completes.  This allows callers to display
    incremental progress while the tests execute.
    """

    def __init__(
        self,
        db: ErrorDB | None = None,
        *,
        graph: KnowledgeGraph | None = None,
        pytest_args: str | None = None,
        workers: int | None = None,
        data_bot: DataBot | None = None,
        result_callback: Callable[[dict[str, Any]], Any] | None = None,
        integration_callback: (
            Callable[[list[str]], dict[str, list[str]] | None] | None
        ) = None,
        disable_auto_integration: bool = False,
        container_image: str = "python:3.11-slim",
        use_container: bool = False,
        container_runtime: str = "docker",
        docker_host: str | None = None,
        container_retries: int = 1,
        container_timeout: float = 300.0,
        history_path: str | Path | None = None,
        state_path: str | Path | None = None,
        metrics_port: int | None = None,
        include_orphans: bool = True,
        discover_orphans: bool = True,
        discover_isolated: bool = True,
        recursive_orphans: bool = getattr(SandboxSettings(), "recursive_orphan_scan", True),
        auto_include_isolated: bool = SandboxSettings().auto_include_isolated,
        recursive_isolated: bool = SandboxSettings().recursive_isolated,
        clean_orphans: bool = False,
        include_redundant: bool = SandboxSettings().test_redundant_modules,
        report_dir: str | Path = Path(settings.self_test_report_dir),
        stub_scenarios: Mapping[str, Any] | None = None,
    ) -> None:
        """Create a new service instance.

        Parameters
        ----------
        container_runtime:
            Executable used to run containers. Can be ``docker`` or ``podman``.
        docker_host:
            Remote host or URL for the container engine. Passed to the runtime
            using ``-H`` for Docker or ``--url`` for Podman.
        metrics_port:
            Port for the Prometheus metrics server. Overrides ``SELF_TEST_METRICS_PORT``.
        integration_callback:
            Callable invoked with a list of successfully tested orphan modules
            after each run. May return a mapping with ``integrated`` and
            ``redundant`` module lists which is aggregated into
            :attr:`integration_details` and exposed through ``results``.
        recursive_orphans:
            When ``True``, follow orphan modules' import chains to include local
            dependencies. Defaults to :class:`~sandbox_settings.SandboxSettings`.
            Set ``SELF_TEST_RECURSIVE_ORPHANS=0`` or pass ``--no-recursive-include``
            to disable.
        auto_include_isolated:
            When ``True``, force discovery of isolated modules and enable
            recursive traversal. Defaults to
            :class:`~sandbox_settings.SandboxSettings`.
        recursive_isolated:
            When ``True``, traverse dependencies of isolated modules.
            Defaults to :class:`~sandbox_settings.SandboxSettings` so both
            sandbox and self-test modes honour the same setting.
        clean_orphans:
            When ``True``, remove successfully integrated modules from
            ``sandbox_data/orphan_modules.json`` after ``integration_callback``
            runs. Can also be enabled via the ``SANDBOX_CLEAN_ORPHANS``
            environment variable.
        include_redundant:
            When ``True``, modules classified as ``redundant`` or ``legacy``
            are still returned from discovery helpers and included in test
            runs. Defaults to
            :class:`~sandbox_settings.SandboxSettings.test_redundant_modules`.
            Can also be enabled via ``SELF_TEST_INCLUDE_REDUNDANT`` or
            ``SANDBOX_TEST_REDUNDANT``.
        """

        self.logger = logging.getLogger(self.__class__.__name__)
        self.graph = graph or KnowledgeGraph()
        self.error_logger = ErrorLogger(db, knowledge_graph=self.graph)
        self.data_bot = data_bot
        self.result_callback = result_callback
        self.container_image = container_image
        self.use_container = use_container
        self.results: dict[str, Any] | None = None
        self.history_path = Path(history_path) if history_path else None
        self._history_db: sqlite3.Connection | None = None
        self._lock_acquired = False
        self.container_runtime = container_runtime
        self.docker_host = docker_host
        self.container_retries = int(os.getenv("SELF_TEST_RETRIES", container_retries))
        try:
            self.container_timeout = float(os.getenv("SELF_TEST_TIMEOUT", str(container_timeout)))
        except ValueError:
            self.container_timeout = container_timeout
        self.offline_install = os.getenv("MENACE_OFFLINE_INSTALL", "0") == "1"
        self.report_dir = Path(report_dir)
        self.image_tar_path = os.getenv("MENACE_SELF_TEST_IMAGE_TAR")
        state_env = os.getenv("SELF_TEST_STATE")
        self.state_path = Path(state_path or state_env) if (state_path or state_env) else None
        env_port = os.getenv("SELF_TEST_METRICS_PORT") if metrics_port is None else None
        if metrics_port is None and env_port is not None:
            try:
                self.metrics_port = int(env_port)
            except ValueError:
                self.metrics_port = None
        else:
            self.metrics_port = metrics_port
        self._metrics_started = False
        self._state: dict[str, Any] | None = None
        if self.state_path and self.state_path.exists():
            try:
                with open(self.state_path, "r", encoding="utf-8") as fh:
                    self._state = json.load(fh) or None
            except (OSError, json.JSONDecodeError):
                self.logger.exception("failed to load state file")
        if self.history_path and self.history_path.suffix == ".db":
            self._history_db = router.get_connection("test_history")
            self._history_db.execute(
                """
                CREATE TABLE IF NOT EXISTS test_history(
                    passed INTEGER,
                    failed INTEGER,
                    coverage REAL,
                    runtime REAL,
                    ts TEXT
                )
                """
            )
            self._history_db.commit()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task | None = None
        self._async_stop: asyncio.Event | None = None
        self._health_server: 'HTTPServer' | None = None
        self._health_thread: threading.Thread | None = None
        self.health_port: int | None = None
        env_args = os.getenv("SELF_TEST_ARGS") if pytest_args is None else pytest_args
        self.pytest_args = shlex.split(env_args) if env_args else []
        env_workers = os.getenv("SELF_TEST_WORKERS") if workers is None else workers
        try:
            self.workers = int(env_workers) if env_workers is not None else 1
        except ValueError:
            self.workers = 1

        # Optional mapping defining scenario templates for generated stubs.
        self.stub_scenarios = dict(stub_scenarios or {})

        # Allow the test runner to be customised via environment variable.  This
        # defaults to ``pytest`` to preserve historical behaviour but can be
        # overridden by setting ``SELF_TEST_RUNNER``.  All discovered modules are
        # executed using this runner within :meth:`_run_once`.
        self.test_runner = os.getenv("SELF_TEST_RUNNER", "pytest")

        disable_env = os.getenv("SELF_TEST_DISABLE_ORPHANS")
        if disable_env is None:
            disable_env = os.getenv("SANDBOX_DISABLE_ORPHANS")
        disable_all = disable_env and disable_env.lower() in ("1", "true", "yes")

        self.include_orphans = bool(include_orphans)
        self.discover_orphans = bool(discover_orphans)
        self.recursive_orphans = bool(recursive_orphans)

        if disable_all:
            self.include_orphans = False
            self.discover_orphans = False
            self.recursive_orphans = False
        else:
            env_orphans = os.getenv("SELF_TEST_INCLUDE_ORPHANS")
            if env_orphans is None:
                env_orphans = os.getenv("SANDBOX_INCLUDE_ORPHANS")
            if env_orphans is not None:
                self.include_orphans = env_orphans.lower() in ("1", "true", "yes")

            env_discover = os.getenv("SELF_TEST_DISCOVER_ORPHANS")
            if env_discover is not None:
                self.discover_orphans = env_discover.lower() in ("1", "true", "yes")

            env_recursive = os.getenv("SELF_TEST_RECURSIVE_ORPHANS")
            if env_recursive is not None:
                self.recursive_orphans = env_recursive.lower() in ("1", "true", "yes")

        self.discover_isolated = bool(discover_isolated)
        self.recursive_isolated = bool(recursive_isolated)
        self.auto_include_isolated = bool(auto_include_isolated)
        if self.auto_include_isolated:
            self.discover_isolated = True
            self.recursive_isolated = True

        self.clean_orphans = bool(clean_orphans)
        env_clean = os.getenv("SANDBOX_CLEAN_ORPHANS")
        if env_clean is not None:
            self.clean_orphans = env_clean.lower() in ("1", "true", "yes")

        self.include_redundant = bool(include_redundant)
        env_redundant = os.getenv("SELF_TEST_INCLUDE_REDUNDANT")
        if env_redundant is None:
            env_redundant = os.getenv("SANDBOX_TEST_REDUNDANT")
        if env_redundant is not None:
            self.include_redundant = env_redundant.lower() in ("1", "true", "yes")

        auto_disable_env = os.getenv("SELF_TEST_DISABLE_AUTO_INTEGRATION")
        if auto_disable_env is not None:
            disable_auto_integration = auto_disable_env.lower() in ("1", "true", "yes")

        if integration_callback is not None:
            self.integration_callback = integration_callback
        elif disable_auto_integration:
            self.integration_callback = None
        else:
            self.integration_callback = self._default_integration

        # populated by ``_discover_orphans`` when recursive orphan discovery is
        # enabled; maps module paths to metadata including import parents and
        # redundancy/legacy classification
        self.orphan_traces: dict[str, dict[str, Any]] = {}

        # aggregated details about the most recent integration run
        self.integration_details: dict[str, list[str]] = {
            "integrated": [],
            "redundant": [],
        }

        # exposed module classification sets for downstream metrics
        self.orphan_passed_modules: list[str] = []
        self.orphan_failed_modules: list[str] = []
        self.orphan_redundant_modules: list[str] = []

        # captured metrics for modules during the most recent run
        self.module_metrics: dict[str, dict[str, Any]] = {}

    def _default_integration(
        self,
        mods: list[str],
        metrics: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, list[str]]:
        """Refresh module map and include ``mods`` into workflows.

        Modules are filtered based on *metrics* using weights and thresholds from
        :class:`~sandbox_settings.SandboxSettings` when provided.

        Returns
        -------
        dict[str, list[str]]
            Mapping containing ``integrated`` and ``redundant`` module lists.
        """

        settings = SandboxSettings()
        if metrics:
            def _score(m: str) -> float:
                data = metrics.get(m, {})
                cov = float(data.get("coverage", 0.0))
                runtime = float(data.get("runtime", 0.0))
                fails = len(data.get("categories", []))
                return (
                    settings.integration_weight_coverage * cov
                    - settings.integration_weight_runtime * runtime
                    - settings.integration_weight_failures * fails
                )

            mods = [m for m in mods if _score(m) >= settings.integration_score_threshold]

        repo = Path(os.getenv("SANDBOX_REPO_PATH", ".")).resolve()
        paths: list[str] = []
        for m in mods:
            p = Path(m)
            try:
                rel = p.resolve().relative_to(repo)
            except Exception:
                rel = p
            paths.append(rel.as_posix())
        paths = sorted(set(paths))
        data_dir = Path(settings.sandbox_data_dir)
        map_file = data_dir / "module_map.json"

        # Ensure the module map knows about the modules before integration.
        try:
            from module_index_db import ModuleIndexDB

            index = ModuleIndexDB(map_file)
            index.refresh(paths, force=True)
            index.save()
        except Exception:
            self.logger.exception("module map refresh failed")

        # Merge passing modules into the sandbox workflows.
        try:
            from sandbox_runner.environment import auto_include_modules

            sig = inspect.signature(auto_include_modules)
            kwargs: dict[str, object] = {}
            if "recursive" in sig.parameters:
                # Always include dependent helpers when merging modules into
                # workflows.
                kwargs["recursive"] = True
            if "validate" in sig.parameters:
                kwargs["validate"] = True
            auto_include_modules(list(paths), **kwargs)
        except Exception:
            self.logger.exception("module auto-inclusion failed")
            if self.clean_orphans:
                try:
                    self._clean_orphan_list(mods, success=False)
                except Exception:
                    self.logger.exception("failed to record orphan modules")
        else:
            if self.clean_orphans:
                try:
                    self._clean_orphan_list(mods, success=True)
                except Exception:
                    self.logger.exception("failed to clean orphan modules")

        return {
            "integrated": paths,
            "redundant": sorted(set(self.orphan_redundant_modules)),
        }

    def get_integration_details(self) -> dict[str, list[str]]:
        """Return aggregated integration results."""

        return {
            "integrated": sorted(set(self.integration_details.get("integrated", []))),
            "redundant": sorted(set(self.integration_details.get("redundant", []))),
        }

    def _store_history(self, rec: dict[str, Any]) -> None:
        if not self.history_path:
            return
        try:
            if self._history_db:
                self._history_db.execute(
                    "INSERT INTO test_history(passed, failed, coverage, runtime, ts) VALUES(?,?,?,?,?)",
                    (
                        int(rec["passed"]),
                        int(rec["failed"]),
                        float(rec["coverage"]),
                        float(rec["runtime"]),
                        rec["ts"],
                    ),
                )
                self._history_db.commit()
            else:
                data = []
                if self.history_path.exists():
                    with open(self.history_path, "r", encoding="utf-8") as fh:
                        try:
                            data = json.load(fh) or []
                        except Exception:
                            data = []
                data.append(rec)
                with open(self.history_path, "w", encoding="utf-8") as fh:
                    json.dump(data, fh)
        except Exception:
            self.logger.exception("failed to store history")

    # ------------------------------------------------------------------
    def _save_state(
        self,
        queue: list[str],
        passed: int,
        failed: int,
        coverage_sum: float,
        runtime: float,
    ) -> None:
        if not self.state_path:
            return
        try:
            data = {
                "queue": queue,
                "passed": passed,
                "failed": failed,
                "coverage_sum": coverage_sum,
                "runtime": runtime,
            }
            with open(self.state_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
        except Exception:
            self.logger.exception("failed to store state")

    # ------------------------------------------------------------------
    def _write_summary_report(self, data: Mapping[str, Any]) -> None:
        """Write *data* to a timestamped JSON file under ``report_dir``."""

        try:
            report_path = Path(self.report_dir)
            report_path.mkdir(parents=True, exist_ok=True)
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            out = report_path / f"report_{ts}.json"
            with open(out, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
        except Exception:
            self.logger.exception("failed to write summary report")

    # ------------------------------------------------------------------
    def _start_health_server(self, port: int) -> None:
        """Launch a minimal HTTP endpoint returning test status."""
        from http.server import BaseHTTPRequestHandler, HTTPServer

        svc = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # type: ignore[override]
                if self.path != "/health":
                    self.send_response(404)
                    self.end_headers()
                    return
                body = json.dumps(
                    {
                        "passed": int(svc.results.get("passed", 0)) if svc.results else 0,
                        "failed": int(svc.results.get("failed", 0)) if svc.results else 0,
                        "runtime": float(svc.results.get("runtime", 0.0)) if svc.results else 0.0,
                        "history": svc.recent_history(),
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: object) -> None:  # pragma: no cover - silence
                return

        server = HTTPServer(("0.0.0.0", port), Handler)
        self.health_port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self._health_server = server
        self._health_thread = thread

    def _stop_health_server(self) -> None:
        if self._health_server:
            try:
                self._health_server.shutdown()
                self._health_server.server_close()
            except Exception:
                self.logger.exception("failed to stop health server")
            if self._health_thread:
                self._health_thread.join(timeout=1.0)
            self._health_server = None
            self._health_thread = None
            self.health_port = None

    def _clear_state(self) -> None:
        if self.state_path and self.state_path.exists():
            try:
                os.unlink(self.state_path)
            except Exception:
                self.logger.exception("failed to delete state file")

    async def _docker_available(self) -> bool:
        """Return ``True`` if the docker CLI is available and acquire the container lock."""
        try:
            await _container_lock.acquire()
            self._lock_acquired = True
            cmd = [self.container_runtime, "--version"]
            if self.docker_host:
                cmd.extend(
                    [
                        "-H" if self.container_runtime == "docker" else "--url",
                        self.docker_host,
                    ]
                )
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
            if proc.returncode != 0:
                _container_lock.release()
                self._lock_acquired = False
            return proc.returncode == 0
        except FileNotFoundError:
            if self._lock_acquired:
                _container_lock.release()
                self._lock_acquired = False
            return False
        except Exception:
            self.logger.exception("docker check failed")
            if self._lock_acquired:
                _container_lock.release()
                self._lock_acquired = False
            return False

    # ------------------------------------------------------------------
    async def _force_remove_container(self, name: str) -> None:
        docker_cmd = [self.container_runtime]
        if self.docker_host:
            docker_cmd.extend([
                "-H" if self.container_runtime == "docker" else "--url",
                self.docker_host,
            ])
        docker_cmd.extend(["rm", "-f", name])
        attempts = self.container_retries + 1
        for attempt in range(attempts):
            try:
                proc = await asyncio.create_subprocess_exec(
                    *docker_cmd,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, err = await asyncio.wait_for(proc.communicate(), timeout=10)
                if proc.returncode == 0:
                    return
                msg = (err.decode().strip() if err else f"code {proc.returncode}")
                self.logger.warning(
                    "failed to remove container %s (attempt %s/%s): %s",
                    name,
                    attempt + 1,
                    attempts,
                    msg,
                )
            except Exception as exc:
                self.logger.warning(
                    "failed to remove container %s (attempt %s/%s): %s",
                    name,
                    attempt + 1,
                    attempts,
                    exc,
                )
            await asyncio.sleep(0.1)
        self.logger.error(
            "could not remove container %s after %s attempts",
            name,
            attempts,
        )
        try:
            self_test_container_failures_total.inc()
        except Exception:
            self.logger.exception("failed to update container failure metric")

    async def _remove_stale_containers(self) -> None:
        docker_cmd = [self.container_runtime]
        if self.docker_host:
            docker_cmd.extend([
                "-H" if self.container_runtime == "docker" else "--url",
                self.docker_host,
            ])
        docker_cmd.extend(["ps", "-aq", "--filter", "label=menace_self_test=1"])
        attempts = self.container_retries + 1
        out: bytes = b""
        for attempt in range(attempts):
            try:
                proc = await asyncio.create_subprocess_exec(
                    *docker_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                out, err = await asyncio.wait_for(proc.communicate(), timeout=10)
                if proc.returncode == 0:
                    break
                self.logger.warning(
                    "failed to list stale containers (attempt %s/%s): %s",
                    attempt + 1,
                    attempts,
                    err.decode().strip() if err else f"code {proc.returncode}",
                )
            except Exception as exc:
                self.logger.warning(
                    "failed to list stale containers (attempt %s/%s): %s",
                    attempt + 1,
                    attempts,
                    exc,
                )
            await asyncio.sleep(0.1)
        else:
            self.logger.error(
                "could not list stale containers after %s attempts", attempts
            )
            try:
                self_test_container_failures_total.inc()
            except Exception:
                self.logger.exception("failed to update container failure metric")
            return

        for cid in out.decode().splitlines():
            cid = cid.strip()
            if cid and all(ch in "0123456789abcdef" for ch in cid.lower()):
                await self._force_remove_container(cid)

    async def _cleanup_containers(self) -> None:
        """Remove containers labelled for self tests and exit."""
        acquired = False
        try:
            _file_lock.acquire()
            acquired = True
            if await self._docker_available():
                await self._remove_stale_containers()
        finally:
            if self._lock_acquired:
                _container_lock.release()
                self._lock_acquired = False
            if acquired:
                try:
                    _file_lock.release()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    def _discover_orphans(self) -> list[str]:
        """Return orphan modules detected in the repository.

        This helper performs discovery only; persisting the results is handled
        by :meth:`_run_once` after combining orphan and isolated modules.
        Discovery always enables recursive dependency tracing so helper modules
        and their parent chains are captured in :attr:`orphan_traces`.
        """
        from sandbox_runner import discover_recursive_orphans as _discover
        from scripts.discover_isolated_modules import (
            discover_isolated_modules as _discover_iso,
        )

        trace = _discover(
            str(Path.cwd()),
            module_map=str(Path(settings.sandbox_data_dir) / "module_map.json"),
        )
        self.orphan_traces = {}
        for k, v in trace.items():
            info: dict[str, Any] = {
                "parents": [
                    str(Path(*p.split(".")).with_suffix(".py"))
                    for p in (
                        v.get("parents") if isinstance(v, dict) else v
                    )
                ],
                "classification": v.get("classification") if isinstance(v, dict) else None,
                "redundant": bool(v.get("redundant")) if isinstance(v, dict) else None,
            }
            self.orphan_traces[str(Path(*k.split(".")).with_suffix(".py"))] = info

        # incorporate isolated modules so dependency chains are complete when
        # either recursive orphan discovery or isolated discovery is requested
        isolated: list[str] = []
        if self.recursive_orphans or self.discover_isolated:
            try:
                isolated = _discover_iso(Path.cwd(), recursive=True)
            except Exception:
                isolated = []
        for m in isolated:
            key = str(Path(m))
            self.orphan_traces.setdefault(
                key,
                {"parents": [], "classification": "candidate", "redundant": False},
            )

        roots = list(self.orphan_traces.keys())
        initial = {m: info.get("parents", []) for m, info in self.orphan_traces.items()}

        def _on_module(rel: str, path: Path, parents: list[str]) -> None:
            entry = self.orphan_traces.setdefault(
                rel,
                {"parents": [], "classification": None, "redundant": None},
            )
            if parents:
                entry["parents"] = list(
                    dict.fromkeys(entry.get("parents", []) + parents)
                )
            if entry.get("classification") is None:
                try:
                    cls = classify_module(path)
                except Exception:  # pragma: no cover - best effort
                    self.logger.exception("classification failed for %s", path)
                    cls = "candidate"
                entry["classification"] = cls
                entry["redundant"] = cls != "candidate"
                if cls in {"legacy", "redundant"}:
                    self.logger.info(
                        "orphan module classified",
                        extra=log_record(module=rel, classification=cls),
                    )

        def _on_dependency(dep_rel: str, _parent_rel: str, chain: list[str]) -> None:
            dep_entry = self.orphan_traces.setdefault(
                dep_rel,
                {"parents": [], "classification": None, "redundant": None},
            )
            dep_entry["parents"] = list(
                dict.fromkeys(dep_entry.get("parents", []) + chain)
            )

        collected = collect_local_dependencies(
            roots,
            initial_parents=initial,
            on_module=_on_module,
            on_dependency=_on_dependency,
        )
        if not collected:
            collected = set(roots)
        modules = [str(Path(p)) for p in sorted(collected)]

        filtered: list[str] = []
        for m in modules:
            p = Path(m)
            info = self.orphan_traces.setdefault(
                m,
                {"parents": [], "classification": None, "redundant": None},
            )
            classification = info.get("classification")
            if classification is None:
                try:
                    classification = classify_module(p)
                except Exception as exc:  # pragma: no cover - best effort
                    self.logger.exception(
                        "classification failed for %s: %s", p, exc
                    )
                    classification = "candidate"
                info["classification"] = classification
                info["redundant"] = classification != "candidate"
            redundant_flag = info.get("redundant")
            if redundant_flag and not self.include_redundant:
                self.logger.info(
                    "%s module skipped",
                    classification,
                    extra=log_record(module=m, classification=classification),
                )
                continue
            passed, warnings, metrics = self._run_module_harness(m)
            info["test_passed"] = passed
            if warnings:
                info["warnings"] = warnings
            if metrics:
                self.module_metrics[m] = metrics
            filtered.append(m)

        return filtered

    # ------------------------------------------------------------------
    def _discover_isolated(self) -> list[str]:
        """Return modules discovered by ``discover_isolated_modules``.

        Persistence of the results is handled by the caller.  This wrapper
        always enables recursive traversal so helper modules are captured and
        parent chains recorded in :attr:`orphan_traces`.
        """
        from scripts.discover_isolated_modules import discover_isolated_modules

        modules = discover_isolated_modules(Path.cwd(), recursive=True)

        roots: list[str] = []
        for m in modules:
            key = str(Path(m))
            entry = self.orphan_traces.setdefault(
                key, {"parents": [], "classification": "candidate", "redundant": False}
            )
            # ``discover_isolated_modules`` only returns non-redundant modules but
            # we record the classification explicitly for later reference.
            roots.append(key)

        # Follow dependency chains so helper modules are recorded alongside the
        # isolated roots while also tracking parent relationships and redundancy
        # flags for all discovered modules.

        initial = {m: self.orphan_traces.get(m, {}).get("parents", []) for m in roots}

        def _on_module(rel: str, path: Path, parents: list[str]) -> None:
            entry = self.orphan_traces.setdefault(
                rel, {"parents": [], "classification": None, "redundant": None}
            )
            if parents:
                entry["parents"] = list(
                    dict.fromkeys(entry.get("parents", []) + parents)
                )
            if entry.get("classification") is None:
                try:
                    cls = classify_module(path)
                except Exception:  # pragma: no cover - best effort
                    self.logger.exception("classification failed for %s", path)
                    cls = "candidate"
                entry["classification"] = cls
                entry["redundant"] = cls != "candidate"
                if cls in {"legacy", "redundant"}:
                    self.logger.info(
                        "orphan module classified",
                        extra=log_record(module=rel, classification=cls),
                    )

        def _on_dependency(dep_rel: str, _parent_rel: str, chain: list[str]) -> None:
            dep_entry = self.orphan_traces.setdefault(
                dep_rel, {"parents": [], "classification": None, "redundant": None}
            )
            dep_entry["parents"] = list(
                dict.fromkeys(dep_entry.get("parents", []) + chain)
            )

        collected = collect_local_dependencies(
            roots,
            initial_parents=initial,
            on_module=_on_module,
            on_dependency=_on_dependency,
        )
        if not collected:
            collected = set(roots)
        discovered = [str(Path(p)) for p in sorted(collected)]

        filtered: list[str] = []
        for m in discovered:
            info = self.orphan_traces.get(m, {})
            if info.get("redundant") and not self.include_redundant:
                cls = info.get("classification", "redundant")
                self.logger.info(
                    "%s module skipped",
                    cls,
                    extra=log_record(module=m, classification=cls),
                )
                continue
            passed, warnings, metrics = self._run_module_harness(m)
            info["test_passed"] = passed
            if warnings:
                info["warnings"] = warnings
            if metrics:
                self.module_metrics[m] = metrics
            filtered.append(m)

        return filtered

    # ------------------------------------------------------------------
    def _clean_orphan_list(self, modules: Iterable[str], *, success: bool = True) -> None:
        """Update ``sandbox_data/orphan_modules.json`` after integration.

        ``modules`` should contain the root modules that were processed.  Any
        helper imports recorded in ``self.orphan_traces`` for those modules are
        included.  When ``success`` is ``True`` the entries are pruned from the
        orphan list; otherwise they are appended with the collected metadata.
        Entries marked ``{"redundant": true}`` are preserved for later
        auditing.
        """
        repo = Path(os.getenv("SANDBOX_REPO_PATH", ".")).resolve()

        def _norm(p: str) -> str:
            q = Path(p)
            if not q.is_absolute():
                q = repo / q
            try:
                return q.resolve().relative_to(repo).as_posix()
            except Exception:
                return str(q)

        roots = {_norm(m) for m in modules}
        targets: set[str] = set(roots)
        for mod, info in self.orphan_traces.items():
            parents = [_norm(p) for p in info.get("parents", [])]
            norm_mod = _norm(mod)
            if norm_mod in roots or any(parent in roots for parent in parents):
                targets.add(norm_mod)

        if success:
            try:
                prune_orphan_cache(repo, targets, self.orphan_traces)
            except Exception:
                self.logger.exception("failed to prune orphan modules")
        else:
            entries: dict[str, dict[str, Any]] = {}
            for mod in targets:
                info = self.orphan_traces.get(mod, {})
                entry: dict[str, Any] = {
                    "parents": [_norm(p) for p in info.get("parents", [])],
                    "classification": info.get("classification", "candidate"),
                    "redundant": bool(info.get("redundant")),
                }
                if "test_passed" in info:
                    entry["test_passed"] = bool(info["test_passed"])
                if info.get("warnings"):
                    entry["warnings"] = info.get("warnings")
                entries[mod] = entry
            try:
                append_orphan_cache(repo, entries)
                append_orphan_classifications(repo, entries)
            except Exception:
                self.logger.exception("failed to append orphan modules")

    # ------------------------------------------------------------------
    def _has_pytest_file(self, mod: str) -> bool:
        """Return ``True`` if a pytest file exists for *mod*.

        The check looks for ``test_<name>.py`` alongside the module as well as
        under a top level ``tests`` directory mirroring the module structure.
        """

        path = Path(mod)
        stem = path.stem
        repo = Path(os.getenv("SANDBOX_REPO_PATH", ".")).resolve()
        candidates = [
            path.parent / f"test_{stem}.py",
            repo / "tests" / f"test_{stem}.py",
            repo / "tests" / path.parent / f"test_{stem}.py",
        ]
        return any(c.exists() for c in candidates)

    # ------------------------------------------------------------------
    def _generate_pytest_stub(
        self,
        mod: str,
        scenarios: Mapping[str, Any] | None = None,
    ) -> Path:
        """Create a temporary pytest stub for *mod* and return its path.

        Parameters
        ----------
        mod:
            Path to the module under test.
        scenarios:
            Optional mapping providing scenario templates and fixture data. The
            structure is ``{"tests": {...}, "fixtures": {...}}`` where the
            ``tests`` mapping associates function or class names with a list of
            scenario dictionaries.  Each scenario may specify ``args``,
            ``kwargs`` and an ``expected`` result.  Arguments can reference
            fixtures via ``{"fixture": "name"}`` entries.
        """

        repo = Path(os.getenv("SANDBOX_REPO_PATH", ".")).resolve()
        stub_root = Path(settings.sandbox_data_dir)
        if not stub_root.is_absolute():
            stub_root = repo / stub_root
        stub_root = stub_root / "selftest_stubs"
        stub_root.mkdir(parents=True, exist_ok=True)
        tmp_dir = Path(tempfile.mkdtemp(prefix="stub_", dir=stub_root))

        try:
            rel = Path(mod).resolve().relative_to(repo)
        except Exception:
            rel = Path(mod).resolve()
        import_name = ".".join(rel.with_suffix("").parts)

        tests: Mapping[str, Any]
        fixtures: Mapping[str, Any]
        if scenarios and ("tests" in scenarios or "fixtures" in scenarios):
            tests = scenarios.get("tests", {})  # type: ignore[assignment]
            fixtures = scenarios.get("fixtures", {})  # type: ignore[assignment]
        else:
            tests = scenarios or {}
            fixtures = {}

        stub_path = tmp_dir / f"test_{Path(mod).stem}_stub.py"
        stub_code = (
            "import importlib, inspect, json\n\n"
            f"SCENARIOS = json.loads('''{json.dumps(tests)}''')\n"
            f"FIXTURES = json.loads('''{json.dumps(fixtures)}''')\n\n"
            "def _resolve(v):\n"
            "    if isinstance(v, dict) and 'fixture' in v:\n"
            "        return FIXTURES.get(v['fixture'])\n"
            "    return v\n\n"
            "def _dummy_value(p):\n"
            "    ann = getattr(p.annotation, '__origin__', p.annotation)\n"
            "    if ann in (int, float, complex):\n"
            "        return 0\n"
            "    if ann is bool:\n"
            "        return False\n"
            "    if ann is str:\n"
            "        return ''\n"
            "    if ann in (list, tuple, set):\n"
            "        return ann()\n"
            "    if ann is dict:\n"
            "        return {}\n"
            "    return None\n\n"
            "def _call_with_dummies(obj, is_class=False):\n"
            "    sig = inspect.signature(obj)\n"
            "    args = []\n"
            "    for i, p in enumerate(sig.parameters.values()):\n"
            "        if is_class and i == 0 and p.name in ('self', 'cls'):\n"
            "            continue\n"
            "        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD) and p.default is inspect._empty:\n"
            "            args.append(_dummy_value(p))\n"
            "    return obj(*args)\n\n"
            "def _run_scenario(func, scen, is_method=False, inst=None):\n"
            "    args = [_resolve(a) for a in scen.get('args', [])]\n"
            "    kwargs = {k: _resolve(v) for k, v in scen.get('kwargs', {}).items()}\n"
            "    target = func if not is_method else getattr(inst, func)\n"
            "    result = target(*args, **kwargs)\n"
            "    if 'expected' in scen:\n"
            "        assert result == _resolve(scen['expected'])\n\n"
            "def test_stub():\n"
            "    try:\n"
            f"        mod = importlib.import_module('{import_name}')\n"
            "    except Exception:\n"
            "        return\n"
            "    for name in dir(mod):\n"
            "        obj = getattr(mod, name)\n"
            "        scen_list = SCENARIOS.get(name, [])\n"
            "        if inspect.isfunction(obj) and obj.__module__ == mod.__name__:\n"
            "            if scen_list:\n"
            "                for scen in scen_list:\n"
            "                    _run_scenario(obj, scen)\n"
            "            else:\n"
            "                try:\n"
            "                    _call_with_dummies(obj)\n"
            "                except Exception:\n"
            "                    pass\n"
            "        elif inspect.isclass(obj) and obj.__module__ == mod.__name__:\n"
            "            if scen_list:\n"
            "                for scen in scen_list:\n"
            "                    init_args = [_resolve(a) for a in scen.get('init_args', [])]\n"
            "                    init_kwargs = {k: _resolve(v) for k, v in scen.get('init_kwargs', {}).items()}\n"
            "                    inst = obj(*init_args, **init_kwargs)\n"
            "                    method = scen.get('method')\n"
            "                    if method:\n"
            "                        _run_scenario(method, scen, True, inst)\n"
            "                    elif 'expected' in scen:\n"
            "                        assert inst == _resolve(scen['expected'])\n"
            "            else:\n"
            "                try:\n"
            "                    _call_with_dummies(obj, True)\n"
            "                except Exception:\n"
            "                    pass\n"
        )
        stub_path.write_text(stub_code, encoding="utf-8")
        return stub_path

    # ------------------------------------------------------------------
    def _run_module_harness(self, mod: str) -> tuple[bool, list[Any], dict[str, Any]]:
        """Run the test harness for *mod* synchronously and return results.

        Returns a tuple ``(passed, warnings, metrics)`` where *passed* is a
        boolean indicating if the harness exited successfully, *warnings* is a
        list of reported warnings and *metrics* contains runtime information
        compatible with :meth:`_test_orphan_modules`.
        """

        stub_path: Path | None = None
        target = mod
        info = self.orphan_traces.get(mod, {})
        if info.get("classification") == "candidate" and not self._has_pytest_file(mod):
            try:
                stub_path = self._generate_pytest_stub(mod, self.stub_scenarios.get(mod))
                target = stub_path.as_posix()
            except Exception:  # pragma: no cover - best effort
                self.logger.exception("failed to create pytest stub for %s", mod)

        cmd = [
            sys.executable,
            "-m",
            self.test_runner,
            "-q",
            "--json-report",
            "--json-report-file=-",
            target,
        ]

        passed = False
        warnings: list[Any] = []
        metrics: dict[str, Any] = {}
        try:
            proc = subprocess.run(cmd, capture_output=True)
            out = proc.stdout.decode() if isinstance(proc.stdout, bytes) else proc.stdout
            report: dict[str, Any] = {}
            try:
                report = json.loads(out or "{}")
            except Exception:  # pragma: no cover - best effort
                self.logger.exception("failed to parse test report for %s", mod)

            summary = report.get("summary", {})
            warnings = report.get("warnings", []) or []
            cov_info = report.get("coverage", {}) or report.get("cov", {}) or {}
            cov = float(
                cov_info.get("percent")
                or cov_info.get("coverage")
                or cov_info.get("percent_covered")
                or 0.0
            )
            runtime = float(
                report.get("duration")
                or summary.get("duration")
                or summary.get("runtime")
                or 0.0
            )
            categories: list[str] = []
            if int(summary.get("failed", 0)):
                categories.append("failed")
            if int(summary.get("error", 0)):
                categories.append("error")

            metrics = {"coverage": cov, "runtime": runtime, "categories": categories}
            passed = proc.returncode == 0
        except Exception:  # pragma: no cover - best effort
            passed = False
        finally:
            if stub_path:
                try:
                    stub_path.unlink()
                    stub_path.parent.rmdir()
                except Exception:
                    pass
        return passed, warnings, metrics

    # ------------------------------------------------------------------
    async def _test_orphan_modules(
        self, paths: Iterable[str]
    ) -> tuple[set[str], set[str], dict[str, dict[str, Any]]]:
        """Execute tests for *paths* and return passing, failing and metrics."""

        passed: set[str] = set()
        failed: set[str] = set()
        metrics: dict[str, dict[str, Any]] = {}
        for mod in paths:
            stub_path: Path | None = None
            target = mod
            info = self.orphan_traces.get(mod, {})
            if info.get("classification") == "candidate" and not self._has_pytest_file(mod):
                try:
                    stub_path = self._generate_pytest_stub(mod, self.stub_scenarios.get(mod))
                    target = stub_path.as_posix()
                except Exception:
                    self.logger.exception("failed to create pytest stub for %s", mod)

            cmd = [
                sys.executable,
                "-m",
                self.test_runner,
                "-q",
                "--json-report",
                "--json-report-file=-",
                target,
            ]
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                out, _ = await proc.communicate()
                report: dict[str, Any] = {}
                try:
                    report = json.loads(out.decode() or "{}")
                except Exception:
                    self.logger.exception("failed to parse test report for %s", mod)

                summary = report.get("summary", {})
                cov_info = report.get("coverage", {}) or report.get("cov", {})
                cov = float(
                    cov_info.get("percent")
                    or cov_info.get("coverage")
                    or cov_info.get("percent_covered")
                    or 0.0
                )
                runtime = float(
                    report.get("duration")
                    or summary.get("duration")
                    or summary.get("runtime")
                    or 0.0
                )
                categories: list[str] = []
                if int(summary.get("failed", 0)):
                    categories.append("failed")
                if int(summary.get("error", 0)):
                    categories.append("error")

                metrics[mod] = {
                    "coverage": cov,
                    "runtime": runtime,
                    "categories": categories,
                }

                if proc.returncode == 0:
                    passed.add(mod)
                else:
                    failed.add(mod)
            except Exception:
                failed.add(mod)
            finally:
                if stub_path:
                    try:
                        stub_path.unlink()
                        stub_path.parent.rmdir()
                    except Exception:
                        pass
        return passed, failed, metrics

    # ------------------------------------------------------------------
    async def _run_once(self, *, refresh_orphans: bool = False) -> None:
        other_args = [a for a in self.pytest_args if a.startswith("-")]
        paths = [a for a in self.pytest_args if not a.startswith("-")]
        if not paths:
            paths = [None]

        env_flags = {
            "SANDBOX_RECURSIVE_ORPHANS": "1" if self.recursive_orphans else "0",
            "SELF_TEST_RECURSIVE_ORPHANS": "1" if self.recursive_orphans else "0",
            "SANDBOX_DISCOVER_ORPHANS": "1" if self.discover_orphans else "0",
            "SELF_TEST_DISCOVER_ORPHANS": "1" if self.discover_orphans else "0",
            "SANDBOX_DISCOVER_ISOLATED": "1" if self.discover_isolated else "0",
            "SELF_TEST_DISCOVER_ISOLATED": "1" if self.discover_isolated else "0",
            "SANDBOX_AUTO_INCLUDE_ISOLATED": "1" if self.auto_include_isolated else "0",
            "SELF_TEST_AUTO_INCLUDE_ISOLATED": "1" if self.auto_include_isolated else "0",
            "SANDBOX_RECURSIVE_ISOLATED": "1" if self.recursive_isolated else "0",
            "SELF_TEST_RECURSIVE_ISOLATED": "1" if self.recursive_isolated else "0",
            "SANDBOX_TEST_REDUNDANT": "1" if self.include_redundant else "0",
            "SELF_TEST_INCLUDE_REDUNDANT": "1" if self.include_redundant else "0",
        }
        original_env = {k: os.environ.get(k) for k in env_flags}
        os.environ.update(env_flags)

        def _restore_env():
            for k, v in original_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        try:
            # reset exposed module sets for each run
            self.orphan_passed_modules = []
            self.orphan_failed_modules = []
            self.orphan_redundant_modules = []
            self.module_metrics = {}
    
            cache_repo = Path.cwd()
            path = cache_repo / Path(settings.sandbox_data_dir) / "orphan_modules.json"
            existing_map = {} if refresh_orphans else load_orphan_cache(cache_repo)
            existing = list(existing_map.keys())
    
            orphan_list: list[str] = existing if self.include_orphans else []
            discovered: list[str] = []
            redundant_list: list[str] = []
    
            if self.include_orphans and (refresh_orphans or not path.exists()):
                try:
                    found = self._discover_orphans()
                    discovered.extend(found)
                    orphan_list.extend(found)
                except Exception:
                    self.logger.exception("failed to discover orphan modules")
    
            if self.discover_orphans or self.discover_isolated:
                try:
                    found = self._discover_orphans()
                    discovered.extend(found)
                    orphan_list.extend(found)
                except Exception:
                    self.logger.exception("failed to auto-discover orphan modules")
    
            if orphan_list:
                orphan_list = list(dict.fromkeys(orphan_list))
    
                initial = {
                    m: self.orphan_traces.get(m, {}).get("parents", [])
                    for m in orphan_list
                }
    
                def _on_module(rel: str, path: Path, parents: list[str]) -> None:
                    entry = self.orphan_traces.setdefault(
                        rel,
                        {"parents": [], "classification": None, "redundant": None},
                    )
                    if parents:
                        entry["parents"] = list(
                            dict.fromkeys(entry.get("parents", []) + parents)
                        )
                    if entry.get("classification") is None:
                        try:
                            cls = classify_module(path)
                        except Exception:  # pragma: no cover - best effort
                            self.logger.exception(
                                "classification failed for %s", path
                            )
                            cls = "candidate"
                        entry["classification"] = cls
                        entry["redundant"] = cls != "candidate"
                        if cls in {"legacy", "redundant"}:
                            self.logger.info(
                                "orphan module classified",
                                extra=log_record(module=rel, classification=cls),
                            )
    
                def _on_dependency(dep_rel: str, _parent_rel: str, chain: list[str]) -> None:
                    dep_entry = self.orphan_traces.setdefault(
                        dep_rel,
                        {"parents": [], "classification": None, "redundant": None},
                    )
                    dep_entry["parents"] = list(
                        dict.fromkeys(dep_entry.get("parents", []) + chain)
                    )
    
                collected = collect_local_dependencies(
                    orphan_list,
                    initial_parents=initial,
                    on_module=_on_module,
                    on_dependency=_on_dependency,
                )
                if not collected:
                    collected = set(orphan_list)
                orphan_list = [str(Path(p)) for p in sorted(collected)]
    
            redundant_list = [
                m for m, info in self.orphan_traces.items() if info.get("redundant")
            ]
            if orphan_list and not self.include_redundant:
                orphan_list = [m for m in orphan_list if m not in redundant_list]
    
            passed_mods: set[str] = set()
            failed_mods: set[str] = set()
            metrics: dict[str, dict[str, Any]] = {}
            if orphan_list:
                passed_mods, failed_mods, metrics = await self._test_orphan_modules(orphan_list)
                self.module_metrics.update(metrics)
                self.orphan_passed_modules = sorted(passed_mods - set(redundant_list))
                self.orphan_failed_modules = sorted(failed_mods)
                self.orphan_redundant_modules = sorted(set(redundant_list))
            try:
                reclassified = {m for m in passed_mods if m in redundant_list}
                legacy_items = [
                    m
                    for m in redundant_list
                    if self.orphan_traces.get(m, {}).get("classification") == "legacy"
                ]
                orphan_modules_tested_total.inc(len(orphan_list))
                orphan_modules_passed_total.inc(len(passed_mods))
                orphan_modules_failed_total.inc(len(failed_mods))
                orphan_modules_reclassified_total.inc(len(reclassified))
                orphan_modules_redundant_total.inc(len(redundant_list))
                orphan_modules_legacy_total.inc(len(legacy_items))
            except Exception:
                pass
    
            combined_file = list(dict.fromkeys(existing + discovered))
            if combined_file or path.exists():
                entries: dict[str, dict[str, Any]] = {}
                for m in combined_file:
                    info = self.orphan_traces.get(m, {})
                    cls = info.get("classification", "candidate")
                    entry = {
                        "parents": info.get("parents", []),
                        "classification": cls,
                        "redundant": cls != "candidate",
                    }
                    if "test_passed" in info:
                        entry["test_passed"] = bool(info["test_passed"])
                    if info.get("warnings"):
                        entry["warnings"] = info.get("warnings")
                    entries[m] = entry
                try:
                    append_orphan_cache(cache_repo, entries)
                    append_orphan_classifications(cache_repo, entries)
                    new_modules = [m for m in combined_file if m not in existing_map]
                    if new_modules:
                        self.logger.info(
                            "Added %d new orphan modules: %s",
                            len(new_modules),
                            ", ".join(sorted(new_modules)),
                        )
                except Exception:
                    self.logger.exception("failed to write orphan modules")
    
            if self._state:
                saved_queue = self._state.get("queue")
                if saved_queue:
                    paths = list(saved_queue)
                passed = int(self._state.get("passed", 0))
                failed = int(self._state.get("failed", 0))
                coverage_total = float(self._state.get("coverage_sum", 0.0))
                runtime_total = float(self._state.get("runtime", 0.0))
            else:
                passed = 0
                failed = 0
                coverage_total = 0.0
                runtime_total = 0.0
    
            self._state = None
    
            all_orphans = set(orphan_list) | set(redundant_list)
            orphan_set: set[str] = set()
    
            queue: list[str] = [
                p or ""
                for p in paths
                if self.include_redundant or not (p and str(Path(p)) in redundant_list)
            ]
            self._save_state(queue, passed, failed, coverage_total, runtime_total)
            proc_info: list[tuple[list[str], str | None, bool, str | None, str]] = []
        except Exception:
            _restore_env()
            raise


        use_container = False
        acquired = False
        try:
            if self.use_container:
                _file_lock.acquire()
                acquired = True
                use_container = await self._docker_available()
            use_pipe = self.result_callback is not None or use_container
            workers_list = [self.workers for _ in paths]
            if use_container and self.workers > 1 and len(paths) > 1:
                base = self.workers // len(paths)
                rem = self.workers % len(paths)
                workers_list = [base + (1 if i < rem else 0) for i in range(len(paths))]
                workers_list = [max(w, 1) for w in workers_list]

            if use_container and self.offline_install and self.image_tar_path:
                docker_cmd = [self.container_runtime]
                if self.docker_host:
                    docker_cmd.extend(
                        [
                            "-H" if self.container_runtime == "docker" else "--url",
                            self.docker_host,
                        ]
                    )
                docker_cmd.extend(["load", "-i", self.image_tar_path])
                try:
                    proc = await asyncio.create_subprocess_exec(
                        *docker_cmd,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                    await proc.wait()
                    if proc.returncode != 0:
                        self.logger.error("docker load failed: %s", self.image_tar_path)
                        use_container = False
                except Exception:
                    self.logger.exception("docker load failed")
                    use_container = False

            for idx, p in enumerate(paths):
                tmp_name: str | None = None
                cmd = [
                    sys.executable,
                    "-m",
                    self.test_runner,
                    "-q",
                    "--json-report",
                ]
                if use_pipe:
                    cmd.append("--json-report-file=-")
                else:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
                    tmp.close()
                    tmp_name = tmp.name
                    cmd.append(f"--json-report-file={tmp_name}")
                cmd.extend(other_args)
                w = workers_list[idx]
                if w > 1:
                    cmd.extend(["-n", str(w)])
                if p:
                    cmd.append(p)

                if use_container:
                    docker_cmd = [self.container_runtime]
                    if self.docker_host:
                        docker_cmd.extend(
                            [
                                "-H" if self.container_runtime == "docker" else "--url",
                                self.docker_host,
                            ]
                        )
                    cname = f"selftest_{uuid.uuid4().hex}"
                    docker_cmd.extend(
                        [
                            "run",
                            "--rm",
                            "--name",
                            cname,
                            "--label",
                            "menace_self_test=1",
                            "-i",
                            "-v",
                            f"{os.getcwd()}:{os.getcwd()}:ro",
                            "-w",
                            os.getcwd(),
                        ]
                    )
                    for k, v in os.environ.items():
                        docker_cmd.extend(["-e", f"{k}={v}"])
                    docker_cmd.append(self.container_image)

                    container_cmd = [
                        "python",
                        "-m",
                        self.test_runner,
                        *cmd[3:],
                    ]

                    full_cmd = [*docker_cmd, *container_cmd]
                    proc_info.append((full_cmd, None, True, cname, p))
                else:
                    proc_info.append((cmd, tmp_name, False, None, p))

            async def _process(
                cmd: list[str], tmp: str | None, is_container: bool, name: str | None
            ) -> tuple[int, int, float, float, bool, str, str, str, list[dict[str, Any]]]:
                report: dict[str, Any] = {}
                out: bytes = b""
                err: bytes = b""
                attempts = self.container_retries + 1 if is_container else 1
                delay = 0.1
                records: list[dict[str, Any]] = []
                for attempt in range(attempts):
                    proc = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    try:
                        out, err = await asyncio.wait_for(
                            proc.communicate(),
                            timeout=self.container_timeout if is_container else None,
                        )
                    except asyncio.TimeoutError as exc:
                        proc.kill()
                        await proc.wait()
                        if is_container and name:
                            await self._force_remove_container(name)
                        try:
                            self_test_container_timeouts_total.inc()
                        except Exception:
                            self.logger.exception("failed to update container timeout metric")
                        rec = log_record(
                            attempt=attempt + 1,
                            error="timeout",
                            name=name,
                        )
                        self.logger.warning("container attempt timed out", extra=rec)
                        records.append(dict(rec))
                        if attempt == attempts - 1:
                            self.logger.error("self test container timed out")
                            break
                        await asyncio.sleep(delay)
                        delay *= 2
                        continue

                    if proc.returncode != 0 and is_container and attempt < attempts - 1:
                        if name:
                            await self._force_remove_container(name)
                        rec = log_record(
                            attempt=attempt + 1,
                            error=f"exit {proc.returncode}",
                            name=name,
                        )
                        self.logger.warning("container attempt failed", extra=rec)
                        records.append(dict(rec))
                        await asyncio.sleep(delay)
                        delay *= 2
                        continue
                    break

                if use_pipe:
                    data = (out or b"") + (err or b"")
                    try:
                        report = json.loads(data.decode() or "{}")
                    except Exception:
                        self.logger.exception("failed to parse test report")
                else:
                    if tmp and os.path.exists(tmp):
                        try:
                            with open(tmp, "r", encoding="utf-8") as fh:
                                report = json.load(fh)
                        except Exception:
                            self.logger.exception("failed to parse test report")

                if tmp:
                    try:
                        os.unlink(tmp)
                    except Exception:
                        pass

                summary = report.get("summary", {})
                pcount = int(summary.get("passed", 0))
                fcount = int(summary.get("failed", 0)) + int(summary.get("error", 0))
                cov_info = report.get("coverage", {}) or report.get("cov", {})
                cov = float(
                    cov_info.get("percent")
                    or cov_info.get("coverage")
                    or cov_info.get("percent_covered")
                    or 0.0
                )
                runtime = float(
                    report.get("duration")
                    or summary.get("duration")
                    or summary.get("runtime")
                    or 0.0
                )

                failed_flag = proc.returncode != 0
                out_snip = (out.decode(errors="ignore") if out else "")[:1000]
                err_snip = (err.decode(errors="ignore") if err else "")[:1000]
                log_snip = ""
                if failed_flag:
                    exc = RuntimeError(
                        f"self tests failed with code {proc.returncode}"
                    )
                    if is_container:
                        cmd_str = " ".join(shlex.quote(c) for c in cmd)
                        err_snip = f"[container {name} cmd: {cmd_str}]\n" + err_snip
                        if name:
                            log_cmd = [self.container_runtime]
                            if self.docker_host:
                                log_cmd.extend([
                                    "-H" if self.container_runtime == "docker" else "--url",
                                    self.docker_host,
                                ])
                            log_cmd.extend(["logs", name])
                            try:
                                lp = await asyncio.create_subprocess_exec(
                                    *log_cmd,
                                    stdout=asyncio.subprocess.PIPE,
                                    stderr=asyncio.subprocess.STDOUT,
                                )
                                lout, _ = await asyncio.wait_for(lp.communicate(), timeout=10)
                                log_snip = (lout.decode(errors="ignore") if lout else "")[:1000]
                            except Exception:
                                log_snip = ""
                        self.logger.error(
                            "self tests failed in container %s: %s (cmd: %s)",
                            name,
                            exc,
                            cmd_str,
                        )
                    else:
                        self.logger.error("self tests failed: %s", exc)
                    try:
                        self.error_logger.log(exc, "self_tests", "sandbox")
                    except Exception:
                        self.logger.exception("error logging failed")
                return (
                    pcount,
                    fcount,
                    cov,
                    runtime,
                    failed_flag,
                    out_snip,
                    err_snip,
                    log_snip,
                    records,
                )

            tasks = [asyncio.create_task(_process(cmd, tmp, is_c, name)) for cmd, tmp, is_c, name, _ in proc_info]

            results = await asyncio.gather(*tasks, return_exceptions=True)
            first_exc: Exception | None = None
            any_failed = False
            stdout_snip = ""
            stderr_snip = ""
            logs_snip = ""
            suite_metrics: dict[str, dict[str, float]] = {}
            retry_errors: dict[str, list[dict[str, Any]]] = {}
            for (cmd, tmp, is_c, name, p), res in zip(proc_info, results):
                if isinstance(res, Exception):
                    if first_exc is None:
                        first_exc = res
                    continue
                (
                    pcount,
                    fcount,
                    cov,
                    runtime,
                    failed_flag,
                    out_snip,
                    err_snip,
                    log_snip,
                    recs,
                ) = res
                key = p or "<root>"
                suite_metrics[key] = {"coverage": cov, "runtime": runtime}
                if recs:
                    retry_errors[key] = recs
                any_failed = any_failed or failed_flag
                passed += pcount
                failed += fcount
                coverage_total += cov
                runtime_total += runtime
                if failed_flag:
                    stdout_snip += out_snip
                    stderr_snip += err_snip
                    logs_snip += log_snip

                if p in queue:
                    queue.remove(p)
                self._save_state(queue, passed, failed, coverage_total, runtime_total)

                if self.result_callback:
                    partial = {
                        "passed": passed,
                        "failed": failed,
                        "coverage": coverage_total / max(len(suite_metrics), 1),
                        "runtime": runtime_total / max(len(suite_metrics), 1),
                    }
                    try:
                        self.result_callback(partial)
                    except Exception:
                        self.logger.exception("result callback failed")

            try:
                self.error_logger.db.add_test_result(passed, failed)
            except Exception:  # pragma: no cover - best effort
                self.logger.exception("failed to store test results")

            coverage = coverage_total / max(len(suite_metrics), 1)
            runtime_avg = runtime_total / max(len(suite_metrics), 1)
            runtime = runtime_avg
            self.results = {
                "passed": passed,
                "failed": failed,
                "coverage": coverage,
                "runtime": runtime_avg,
                "total_runtime": runtime_total,
                "suite_metrics": suite_metrics,
            }
            if retry_errors:
                self.results["retry_errors"] = retry_errors
            self.results["module_metrics"] = self.module_metrics
            failed_modules = {
                Path(m).stem.lower()
                for m, info in self.module_metrics.items()
                if "failed" in info.get("categories", [])
                or "error" in info.get("categories", [])
            }
            set_failed_critical_tests(failed_modules)
            passed_set: list[str] = []
            if all_orphans:
                passed_set = list(self.orphan_passed_modules)
                orphan_failed = list(self.orphan_failed_modules)
                redundant_set = list(self.orphan_redundant_modules)
                self.results["orphan_total"] = len(all_orphans)
                self.results["orphan_failed"] = len(orphan_failed)
                self.results["orphan_passed"] = self.orphan_passed_modules
                self.results["orphan_passed_modules"] = self.orphan_passed_modules
                self.results["orphan_redundant"] = self.orphan_redundant_modules
                self.results["orphan_redundant_modules"] = self.orphan_redundant_modules
                self.results["orphan_failed_modules"] = self.orphan_failed_modules
            if stdout_snip or stderr_snip or logs_snip:
                self.results["stdout"] = stdout_snip
                self.results["stderr"] = stderr_snip
                if logs_snip:
                    self.results["logs"] = logs_snip
            self._store_history(
                {
                    "passed": passed,
                    "failed": failed,
                    "coverage": coverage,
                    "runtime": runtime,
                    "ts": datetime.utcnow().isoformat(),
                }
            )

            if self.result_callback:
                try:
                    self.result_callback(self.results)
                except Exception:
                    self.logger.exception("result callback failed")
            if self.clean_orphans and passed_set:
                try:
                    self._clean_orphan_list(passed_set, success=False)
                except Exception:
                    self.logger.exception("failed to clean orphan list")

            if passed_set:
                integrate_mods = [
                    m for m in passed_set if not self.orphan_traces.get(m, {}).get("redundant")
                ]

                if integrate_mods:
                    initial = {
                        m: self.orphan_traces.get(m, {}).get("parents", [])
                        for m in integrate_mods
                    }

                    def _on_module(rel: str, path: Path, parents: list[str]) -> None:
                        entry = self.orphan_traces.setdefault(
                            rel,
                            {"parents": [], "classification": None, "redundant": None},
                        )
                        if parents:
                            entry["parents"] = list(
                                dict.fromkeys(entry.get("parents", []) + parents)
                            )
                        if entry.get("classification") is None:
                            try:
                                cls = classify_module(path)
                            except Exception:  # pragma: no cover - best effort
                                self.logger.exception(
                                    "classification failed for %s", path
                                )
                                cls = "candidate"
                            entry["classification"] = cls
                            entry["redundant"] = cls != "candidate"

                    def _on_dep(dep_rel: str, _parent: str, chain: list[str]) -> None:
                        dep_entry = self.orphan_traces.setdefault(
                            dep_rel,
                            {"parents": [], "classification": None, "redundant": None},
                        )
                        dep_entry["parents"] = list(
                            dict.fromkeys(dep_entry.get("parents", []) + chain)
                        )

                    collected = collect_local_dependencies(
                        integrate_mods,
                        initial_parents=initial,
                        on_module=_on_module,
                        on_dependency=_on_dep,
                    )
                    if collected:
                        integrate_mods = [str(Path(p)) for p in sorted(collected)]

                cleaned = False
                info: dict[str, list[str]] | None = None
                integrated_now = 0
                if self.integration_callback and integrate_mods:
                    try:
                        metric_subset = {
                            m: self.module_metrics.get(m, {}) for m in integrate_mods
                        }
                        sig = inspect.signature(self.integration_callback)
                        if len(sig.parameters) > 1:
                            info = self.integration_callback(integrate_mods, metric_subset)
                        else:
                            info = self.integration_callback(integrate_mods)
                    except Exception:
                        self.logger.exception("orphan integration failed")
                    else:
                        cleaned = self.integration_callback is self._default_integration
                        if isinstance(info, dict):
                            integrated = info.get("integrated", integrate_mods)
                            self.integration_details["integrated"].extend(integrated)
                            self.integration_details["redundant"].extend(
                                info.get("redundant", [])
                            )
                            integrated_now = len(integrated)
                        else:
                            self.integration_details["integrated"].extend(integrate_mods)
                            integrated_now = len(integrate_mods)
                elif integrate_mods:
                    # No callback but still record integrated modules
                    self.integration_details["integrated"].extend(integrate_mods)
                    integrated_now = len(integrate_mods)

                if self.clean_orphans and integrate_mods and not cleaned:
                    try:
                        self._clean_orphan_list(integrate_mods, success=integrated_now > 0)
                    except Exception:
                        self.logger.exception("failed to clean orphan list")

                if self.orphan_redundant_modules:
                    self.integration_details["redundant"].extend(
                        self.orphan_redundant_modules
                    )

                try:
                    orphan_modules_reintroduced_total.inc(integrated_now)
                except Exception:
                    pass

            if self.results is not None:
                self.results["integration"] = self.get_integration_details()

            if not queue:
                self._clear_state()

            if self.data_bot:
                try:
                    self.data_bot.db.log_eval("self_tests", "coverage", float(coverage))
                    self.data_bot.db.log_eval("self_tests", "runtime", float(runtime))
                except Exception:
                    self.logger.exception("failed to store metrics")

            try:
                self_test_passed_total.set(float(passed))
                self_test_failed_total.set(float(failed))
                self_test_average_runtime_seconds.set(float(runtime))
                self_test_average_coverage.set(float(coverage))
            except Exception:
                self.logger.exception("failed to update metrics")

            if (first_exc or any_failed) and self.results:
                try:
                    self._write_summary_report(self.results)
                except Exception:
                    self.logger.exception("failed to persist summary report")

            if first_exc:
                raise first_exc
            if any_failed:
                raise RuntimeError("self tests failed")
        finally:
            if self._lock_acquired:
                _container_lock.release()
                self._lock_acquired = False
            if acquired:
                try:
                    _file_lock.release()
                except Exception:
                    pass

            _restore_env()
    async def _schedule_loop(self, interval: float, *, refresh_orphans: bool = False) -> None:
        assert self._async_stop is not None
        while not self._async_stop.is_set():
            try:
                if refresh_orphans:
                    await self._run_once(refresh_orphans=refresh_orphans)
                else:
                    await self._run_once()
            except Exception:
                self.logger.exception("self test run failed")
            try:
                await asyncio.wait_for(self._async_stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass

    # ------------------------------------------------------------------
    def recent_history(self, limit: int = 10) -> list[dict[str, Any]]:
        if not self.history_path:
            return []
        try:
            if self._history_db:
                cur = self._history_db.execute(
                    "SELECT passed, failed, coverage, runtime, ts FROM test_history ORDER BY ts DESC LIMIT ?",
                    (limit,),
                )
                rows = cur.fetchall()
                return [
                    {
                        "passed": int(r[0]),
                        "failed": int(r[1]),
                        "coverage": float(r[2]),
                        "runtime": float(r[3]),
                        "ts": r[4],
                    }
                    for r in rows
                ]
            else:
                if not self.history_path.exists():
                    return []
                with open(self.history_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh) or []
                return list(reversed(data))[:limit]
        except Exception:
            self.logger.exception("failed to read history")
            return []

    # ------------------------------------------------------------------
    def run_once(
        self, *, refresh_orphans: bool = False
    ) -> tuple[dict[str, Any], list[str]]:
        """Execute the self tests once and return results and passing modules.

        Any exception raised by :meth:`_run_once` is logged and swallowed.
        """

        if self.metrics_port is not None and not self._metrics_started:
            try:
                _me.start_metrics_server(int(self.metrics_port))
                self._metrics_started = True
            except Exception:
                self.logger.exception("failed to start metrics server")

        try:
            asyncio.run(self._run_once(refresh_orphans=refresh_orphans))
        except Exception:
            self.logger.exception("self test run failed")
        finally:
            if self._metrics_started:
                _me.stop_metrics_server()
                self._metrics_started = False

        # Collect orphan metrics for future reference
        if self.results is not None:
            try:
                self._record_orphan_results()
            except Exception:
                # best effort – failures here should not break callers
                self.logger.exception("failed to record orphan results")

        passed_modules: list[str] = []
        if self.results is not None:
            passed_modules = list(self.results.get("orphan_passed_modules", []))
        return self.results or {}, passed_modules

    # ------------------------------------------------------------------
    def _record_orphan_results(self) -> None:
        """Persist metrics and classifications for tested orphan modules."""

        modules = set(self.orphan_passed_modules) | set(self.orphan_failed_modules)
        if not modules:
            return

        repo = Path(os.getenv("SANDBOX_REPO_PATH", "."))
        data_dir = Path(settings.sandbox_data_dir)
        if not data_dir.is_absolute():
            data_dir = repo / data_dir
        result_path = data_dir / "orphan_results.json"

        roi_map: dict[str, float] = {}
        try:  # pragma: no cover - optional dependency
            from roi_tracker import ROITracker  # type: ignore

            tracker = ROITracker()
            roi_file = data_dir / "roi_history.json"
            if roi_file.exists():
                tracker.load_history(str(roi_file))
                roi_map = {
                    m: float(sum(tracker.module_deltas.get(m, []))) for m in modules
                }
        except Exception:
            roi_map = {}

        results: dict[str, dict[str, Any]] = {}
        for m in modules:
            info = self.module_metrics.get(m, {})
            classification = self.orphan_traces.get(m, {}).get("classification")
            results[m] = {
                "coverage": float(info.get("coverage", 0.0)),
                "roi_delta": float(roi_map.get(m, 0.0)),
                "classification": classification,
            }

        self.results.setdefault("orphan_results", {}).update(results)

        try:
            existing = (
                json.loads(result_path.read_text()) if result_path.exists() else {}
            )
            if not isinstance(existing, dict):
                existing = {}
        except Exception:
            existing = {}
        existing.update(results)
        try:
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(json.dumps(existing, indent=2))
        except Exception:  # pragma: no cover - best effort
            pass

    # ------------------------------------------------------------------
    @staticmethod
    def orphan_summary(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
        """Return stored orphan testing metrics."""

        repo = Path(os.getenv("SANDBOX_REPO_PATH", "."))
        data_dir = Path(settings.sandbox_data_dir)
        if not data_dir.is_absolute():
            data_dir = repo / data_dir
        target = Path(path) if path else data_dir / "orphan_results.json"
        try:
            data = json.loads(target.read_text()) if target.exists() else {}
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return {}

    # ------------------------------------------------------------------
    def run_continuous(
        self,
        interval: float = 86400.0,
        *,
        loop: asyncio.AbstractEventLoop | None = None,
        health_port: int | None = None,
        refresh_orphans: bool = False,
    ) -> asyncio.Task:
        """Start the background schedule loop on *loop*."""

        if self._task:
            return self._task
        self._loop = loop or asyncio.get_event_loop()
        self._async_stop = asyncio.Event()
        if health_port is not None:
            try:
                self._start_health_server(int(health_port))
            except Exception:
                self.logger.exception("failed to start health server")
        if self.metrics_port is not None and not self._metrics_started:
            try:
                _me.start_metrics_server(int(self.metrics_port))
                self._metrics_started = True
            except Exception:
                self.logger.exception("failed to start metrics server")
        self._task = self._loop.create_task(self._schedule_loop(interval, refresh_orphans=refresh_orphans))
        return self._task

    # ------------------------------------------------------------------
    def run_scheduled(
        self,
        interval: float = 86400.0,
        *,
        runs: int | None = None,
        refresh_orphans: bool = False,
    ) -> None:
        """Run :meth:`_run_once` repeatedly with a delay between runs."""

        if self.metrics_port is not None and not self._metrics_started:
            try:
                _me.start_metrics_server(int(self.metrics_port))
                self._metrics_started = True
            except Exception:
                self.logger.exception("failed to start metrics server")
        count = 0
        while True:
            try:
                sig = inspect.signature(self._run_once)
                kwargs = (
                    {"refresh_orphans": refresh_orphans}
                    if "refresh_orphans" in sig.parameters
                    else {}
                )
                asyncio.run(self._run_once(**kwargs))
            except Exception:
                self.logger.exception("self test run failed")
            count += 1
            if runs is not None and count >= runs:
                break
            time.sleep(interval)
        if self._metrics_started:
            _me.stop_metrics_server()
            self._metrics_started = False

    # ------------------------------------------------------------------
    async def stop(self) -> None:
        """Stop the schedule loop and wait for completion."""

        if not self._task:
            return
        assert self._async_stop is not None
        self._async_stop.set()
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        finally:
            self._task = None
            self._stop_health_server()
            if self._metrics_started:
                _me.stop_metrics_server()
                self._metrics_started = False


__all__ = [
    "SelfTestService",
    "self_test_passed_total",
    "self_test_failed_total",
    "self_test_average_runtime_seconds",
    "self_test_average_coverage",
    "self_test_container_failures_total",
    "self_test_container_timeouts_total",
    "cli",
    "main",
]


def cli(argv: list[str] | None = None) -> int:
    """Command line interface for running the self tests."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run self tests once", aliases=["manual"])
    run.add_argument("paths", nargs="*", help="Test paths or patterns")
    run.add_argument("--workers", type=int, default=1, help="Number of pytest workers")
    run.add_argument(
        "--container-image",
        default="python:3.11-slim",
        help="Docker image when using containers",
    )
    run.add_argument(
        "--container-runtime",
        default="docker",
        help="Container runtime executable",
    )
    run.add_argument(
        "--docker-host",
        help="Docker/Podman host or URL",
    )
    run.add_argument(
        "--use-container",
        action="store_true",
        help="Execute tests inside a Docker container",
    )
    run.add_argument(
        "--history",
        help="Path to JSON/DB file storing run history",
    )
    run.add_argument(
        "--state",
        help="Path to JSON file storing current run state",
    )
    run.add_argument(
        "--pytest-args",
        default=None,
        help="Additional arguments passed to pytest",
    )
    run.add_argument(
        "--retries",
        type=int,
        default=1,
        help="Container retry attempts on failure",
    )
    run.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Container timeout in seconds",
    )
    run.add_argument(
        "--metrics-port",
        type=int,
        help="Port to expose Prometheus gauges",
    )
    run.add_argument(
        "--report-dir",
        default=settings.self_test_report_dir,
        help="Directory to store failure reports",
    )
    run.add_argument(
        "--include-orphans",
        action="store_true",
        help=(
            f"Also test modules listed in "
            f"{Path(settings.sandbox_data_dir) / 'orphan_modules.json'}"
        ),
    )
    run.add_argument(
        "--discover-orphans",
        action="store_true",
        help="Automatically run find_orphan_modules and include results",
    )
    run.add_argument(
        "--auto-include-isolated",
        dest="discover_isolated",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Automatically run discover_isolated_modules and append results "
            "(sets SANDBOX_AUTO_INCLUDE_ISOLATED=1 and SANDBOX_RECURSIVE_ISOLATED=1)"
        ),
    )
    run.add_argument(
        "--refresh-orphans",
        action="store_true",
        help="Regenerate orphan list before running",
    )
    run.add_argument(
        "--recursive-include",
        "--recursive-orphans",
        dest="recursive_orphans",
        action="store_true",
        default=True,
        help=(
            "Recursively discover dependent orphan chains (sets "
            "SANDBOX_RECURSIVE_ORPHANS=1; alias: --recursive-orphans)"
        ),
    )
    run.add_argument(
        "--no-recursive-include",
        "--no-recursive-orphans",
        dest="recursive_orphans",
        action="store_false",
        help=(
            "Do not recurse through orphan dependencies (sets "
            "SANDBOX_RECURSIVE_ORPHANS=0)"
        ),
    )
    run.add_argument(
        "--recursive-isolated",
        dest="recursive_isolated",
        action="store_true",
        default=get_recursive_isolated(),
        help="Recursively discover dependencies of isolated modules (default: enabled)",
    )
    run.add_argument(
        "--no-recursive-isolated",
        dest="recursive_isolated",
        action="store_false",
        help="Do not recurse through isolated module dependencies",
    )
    run.add_argument(
        "--clean-orphans",
        action="store_true",
        help="Remove passing modules from orphan_modules.json",
    )
    run.add_argument(
        "--include-redundant",
        "--test-redundant",
        dest="include_redundant",
        action="store_true",
        help=(
            "Also run tests for modules classified as redundant or legacy "
            "(sets SELF_TEST_INCLUDE_REDUNDANT=1 and SANDBOX_TEST_REDUNDANT=1)"
        ),
    )

    sched = sub.add_parser("run-scheduled", help="Run self tests on an interval")
    sched.add_argument("paths", nargs="*", help="Test paths or patterns")
    sched.add_argument("--interval", type=float, default=86400.0, help="Run interval in seconds")
    sched.add_argument("--workers", type=int, default=1, help="Number of pytest workers")
    sched.add_argument(
        "--container-image",
        default="python:3.11-slim",
        help="Docker image when using containers",
    )
    sched.add_argument(
        "--container-runtime",
        default="docker",
        help="Container runtime executable",
    )
    sched.add_argument(
        "--docker-host",
        help="Docker/Podman host or URL",
    )
    sched.add_argument(
        "--history",
        help="Path to JSON/DB file storing run history",
    )
    sched.add_argument(
        "--state",
        help="Path to JSON file storing current run state",
    )
    sched.add_argument(
        "--pytest-args",
        default=None,
        help="Additional arguments passed to pytest",
    )
    sched.add_argument(
        "--retries",
        type=int,
        default=1,
        help="Container retry attempts on failure",
    )
    sched.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Container timeout in seconds",
    )
    sched.add_argument(
        "--metrics-port",
        type=int,
        help="Port to expose Prometheus gauges",
    )
    sched.add_argument(
        "--report-dir",
        default=settings.self_test_report_dir,
        help="Directory to store failure reports",
    )
    sched.add_argument(
        "--include-orphans",
        action="store_true",
        help=(
            f"Also test modules listed in "
            f"{Path(settings.sandbox_data_dir) / 'orphan_modules.json'}"
        ),
    )
    sched.add_argument(
        "--discover-orphans",
        action="store_true",
        help="Automatically run find_orphan_modules and include results",
    )
    sched.add_argument(
        "--auto-include-isolated",
        dest="discover_isolated",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically run discover_isolated_modules and append results",
    )
    sched.add_argument(
        "--refresh-orphans",
        action="store_true",
        help="Regenerate orphan list before running",
    )
    sched.add_argument(
        "--recursive-include",
        "--recursive-orphans",
        dest="recursive_orphans",
        action="store_true",
        default=True,
        help="Recursively discover dependent orphan chains (alias: --recursive-orphans)",
    )
    sched.add_argument(
        "--no-recursive-include",
        "--no-recursive-orphans",
        dest="recursive_orphans",
        action="store_false",
        help="Do not recurse through orphan dependencies",
    )
    sched.add_argument(
        "--recursive-isolated",
        dest="recursive_isolated",
        action="store_true",
        default=get_recursive_isolated(),
        help="Recursively discover dependencies of isolated modules (default: enabled)",
    )
    sched.add_argument(
        "--no-recursive-isolated",
        dest="recursive_isolated",
        action="store_false",
        help="Do not recurse through isolated module dependencies",
    )
    sched.add_argument(
        "--clean-orphans",
        action="store_true",
        help="Remove passing modules from orphan_modules.json",
    )
    sched.add_argument(
        "--include-redundant",
        "--test-redundant",
        dest="include_redundant",
        action="store_true",
        help=(
            "Also run tests for modules classified as redundant or legacy "
            "(sets SELF_TEST_INCLUDE_REDUNDANT=1 and SANDBOX_TEST_REDUNDANT=1)"
        ),
    )
    sched.add_argument(
        "--no-container",
        dest="use_container",
        action="store_false",
        help="Run tests on the host instead of a container",
    )
    sched.set_defaults(use_container=True)

    rep = sub.add_parser("report", help="Show last self test report")
    rep.add_argument(
        "--report-dir",
        default=settings.self_test_report_dir,
        help="Directory containing failure reports",
    )

    clean = sub.add_parser("cleanup", help="Remove stale test containers")
    clean.add_argument(
        "--container-runtime",
        default="docker",
        help="Container runtime executable",
    )
    clean.add_argument(
        "--docker-host",
        help="Docker/Podman host or URL",
    )
    clean.add_argument(
        "--retries",
        type=int,
        default=1,
        help="Container retry attempts on failure",
    )

    args = parser.parse_args(argv)

    rec_flag = getattr(args, "recursive_orphans", True)
    val = "1" if rec_flag else "0"
    os.environ["SANDBOX_RECURSIVE_ORPHANS"] = val
    os.environ["SELF_TEST_RECURSIVE_ORPHANS"] = val
    recursive_orphans = rec_flag

    os.environ["SANDBOX_DISCOVER_ORPHANS"] = (
        "1" if getattr(args, "discover_orphans", False) else "0"
    )
    os.environ["SELF_TEST_DISCOVER_ORPHANS"] = (
        "1" if getattr(args, "discover_orphans", False) else "0"
    )

    if getattr(args, "discover_isolated", False):
        set_recursive_isolated(True)
        os.environ["SANDBOX_DISCOVER_ISOLATED"] = "1"
        os.environ["SELF_TEST_DISCOVER_ISOLATED"] = "1"
    else:
        set_recursive_isolated(getattr(args, "recursive_isolated", False))
        os.environ["SANDBOX_DISCOVER_ISOLATED"] = "0"
        os.environ["SELF_TEST_DISCOVER_ISOLATED"] = "0"

    auto_iso = "1" if getattr(args, "discover_isolated", False) else "0"
    os.environ["SANDBOX_AUTO_INCLUDE_ISOLATED"] = auto_iso
    os.environ["SELF_TEST_AUTO_INCLUDE_ISOLATED"] = auto_iso

    os.environ["SANDBOX_TEST_REDUNDANT"] = (
        "1" if getattr(args, "include_redundant", False) else "0"
    )
    os.environ["SELF_TEST_INCLUDE_REDUNDANT"] = (
        "1" if getattr(args, "include_redundant", False) else "0"
    )

    if args.cmd == "run":
        pytest_args = []
        if args.pytest_args:
            pytest_args.extend(shlex.split(args.pytest_args))
        if args.paths:
            pytest_args.extend(args.paths)
        service = SelfTestService(
            pytest_args=" ".join(pytest_args) if pytest_args else None,
            workers=args.workers,
            container_image=args.container_image,
            container_runtime=args.container_runtime,
            docker_host=args.docker_host,
            use_container=args.use_container,
            history_path=args.history,
            state_path=args.state,
            container_retries=args.retries,
            container_timeout=args.timeout,
            metrics_port=args.metrics_port,
            include_orphans=args.include_orphans,
            discover_orphans=args.discover_orphans,
            discover_isolated=args.discover_isolated,
            recursive_orphans=recursive_orphans,
            recursive_isolated=args.recursive_isolated,
            auto_include_isolated=args.discover_isolated,
            clean_orphans=args.clean_orphans,
            include_redundant=args.include_redundant,
            report_dir=args.report_dir,
        )
        try:
            asyncio.run(service._run_once(refresh_orphans=args.refresh_orphans))
        except Exception as exc:
            print(f"self test run failed: {exc}", file=sys.stderr)
            if service.results:
                out = service.results.get("stdout")
                err = service.results.get("stderr")
                if out:
                    print(out, file=sys.stderr)
                if err:
                    print(err, file=sys.stderr)
            return 1
        return 0

    if args.cmd == "run-scheduled":
        pytest_args = []
        if args.pytest_args:
            pytest_args.extend(shlex.split(args.pytest_args))
        if args.paths:
            pytest_args.extend(args.paths)
        service = SelfTestService(
            pytest_args=" ".join(pytest_args) if pytest_args else None,
            workers=args.workers,
            container_image=args.container_image,
            container_runtime=args.container_runtime,
            docker_host=args.docker_host,
            use_container=args.use_container,
            history_path=args.history,
            state_path=args.state,
            container_retries=args.retries,
            container_timeout=args.timeout,
            metrics_port=args.metrics_port,
            include_orphans=args.include_orphans,
            discover_orphans=args.discover_orphans,
            discover_isolated=args.discover_isolated,
            recursive_orphans=recursive_orphans,
            recursive_isolated=args.recursive_isolated,
            auto_include_isolated=args.discover_isolated,
            clean_orphans=args.clean_orphans,
            include_redundant=args.include_redundant,
            report_dir=args.report_dir,
        )
        try:
            service.run_scheduled(interval=args.interval, refresh_orphans=args.refresh_orphans)
        except KeyboardInterrupt:
            pass
        return 0

    if args.cmd == "report":
        report_dir = Path(args.report_dir)
        files = sorted(report_dir.glob("*.json"))
        if not files:
            print("no reports found", file=sys.stderr)
            return 1
        print(files[-1].read_text())
        return 0

    if args.cmd == "cleanup":
        service = SelfTestService(
            container_runtime=args.container_runtime,
            docker_host=args.docker_host,
            container_retries=args.retries,
        )
        try:
            asyncio.run(service._cleanup_containers())
        except Exception as exc:
            print(f"cleanup failed: {exc}", file=sys.stderr)
            return 1
        return 0

    parser.error("unknown command")
    return 1


def main(argv: list[str] | None = None) -> None:
    sys.exit(cli(argv))


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
