from __future__ import annotations

"""Service running self tests on a schedule."""

import asyncio
import time
from asyncio import Lock
from filelock import FileLock
import json
import logging
import ast
import os
import shlex
import sqlite3
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable
import threading
import inspect

from orphan_analyzer import analyze_redundancy

if os.getenv("SANDBOX_CENTRAL_LOGGING") == "1":
    from logging_utils import setup_logging

    setup_logging()

try:
    from .data_bot import DataBot
    from .error_bot import ErrorDB
    from .error_logger import ErrorLogger
except Exception:  # pragma: no cover - fallback when imported directly
    from data_bot import DataBot  # type: ignore
    from error_bot import ErrorDB  # type: ignore
    from error_logger import ErrorLogger  # type: ignore

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
_file_lock = FileLock(os.getenv("SELF_TEST_LOCK_FILE", "sandbox_data/self_test.lock"))


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
        pytest_args: str | None = None,
        workers: int | None = None,
        data_bot: DataBot | None = None,
        result_callback: Callable[[dict[str, Any]], Any] | None = None,
        integration_callback: Callable[[list[str]], None] | None = None,
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
        recursive_orphans: bool = True,
        recursive_isolated: bool | None = None,
        clean_orphans: bool = False,
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
            after each run. Can be used to merge them into the sandbox's module
            map.
        recursive_orphans:
            When ``True``, follow orphan modules' import chains to include local
            dependencies. Defaults to ``True``. Set ``SELF_TEST_RECURSIVE_ORPHANS=0``
            or pass ``--no-recursive-include`` to disable.
        recursive_isolated:
            When ``True``, traverse dependencies of isolated modules. Defaults
            to :func:`get_recursive_isolated` so both sandbox and self-test
            modes honour the ``SANDBOX_RECURSIVE_ISOLATED``/
            ``SELF_TEST_RECURSIVE_ISOLATED`` flags.
        clean_orphans:
            When ``True``, remove successfully integrated modules from
            ``sandbox_data/orphan_modules.json`` after ``integration_callback``
            runs. Can also be enabled via the ``SANDBOX_CLEAN_ORPHANS``
            environment variable.
        """

        self.logger = logging.getLogger(self.__class__.__name__)
        self.error_logger = ErrorLogger(db)
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
            except Exception:
                self.logger.exception("failed to load state file")
        if self.history_path and self.history_path.suffix == ".db":
            self._history_db = sqlite3.connect(self.history_path)
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
            if env_recursive is None:
                env_recursive = os.getenv("SANDBOX_RECURSIVE_ORPHANS")
            if env_recursive is not None:
                self.recursive_orphans = env_recursive.lower() in ("1", "true", "yes")

        self.discover_isolated = bool(discover_isolated)
        env_isolated = (
            os.getenv("SELF_TEST_AUTO_INCLUDE_ISOLATED")
            or os.getenv("SANDBOX_AUTO_INCLUDE_ISOLATED")
            or os.getenv("SELF_TEST_DISCOVER_ISOLATED")
            or os.getenv("SANDBOX_DISCOVER_ISOLATED")
        )
        if env_isolated is not None:
            self.discover_isolated = env_isolated.lower() in ("1", "true", "yes")
        self.recursive_isolated = (
            get_recursive_isolated()
            if recursive_isolated is None
            else bool(recursive_isolated)
        )
        auto_inc = (
            os.getenv("SELF_TEST_AUTO_INCLUDE_ISOLATED")
            or os.getenv("SANDBOX_AUTO_INCLUDE_ISOLATED")
        )
        if auto_inc and auto_inc.lower() in ("1", "true", "yes"):
            self.discover_isolated = True
            self.recursive_isolated = True

        self.clean_orphans = bool(clean_orphans)
        env_clean = os.getenv("SANDBOX_CLEAN_ORPHANS")
        if env_clean is not None:
            self.clean_orphans = env_clean.lower() in ("1", "true", "yes")

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
        # redundancy classification
        self.orphan_traces: dict[str, dict[str, Any]] = {}

    def _default_integration(self, mods: list[str]) -> None:
        """Refresh module map and include ``mods`` into workflows."""

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
        data_dir = Path(os.getenv("SANDBOX_DATA_DIR", "sandbox_data"))
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
        else:
            if self.clean_orphans:
                try:
                    self._clean_orphan_list(mods)
                except Exception:
                    self.logger.exception("failed to clean orphan modules")

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
    def _collect_recursive(self, mods: Iterable[str]) -> set[str]:
        """Return modules reachable from *mods* while recording parent chains.

        ``self.orphan_traces`` is updated in-place to ensure each discovered
        module contains a ``parents`` list preserving the import chain and a
        ``redundant`` flag determined via :func:`analyze_redundancy`.
        """

        repo = Path(os.getenv("SANDBOX_REPO_PATH", "."))
        queue: list[tuple[Path, list[str]]] = []
        seen: set[str] = set()

        for m in mods:
            p = Path(m)
            if not p.is_absolute():
                p = repo / p
            key = p.relative_to(repo).as_posix()
            parents = list(self.orphan_traces.get(key, {}).get("parents", []))
            queue.append((p, parents))

        while queue:
            path, parents = queue.pop()
            try:
                rel = path.relative_to(repo).as_posix()
            except Exception:
                continue

            entry = self.orphan_traces.setdefault(rel, {"parents": [], "redundant": None})
            if parents:
                entry_parents = entry.get("parents", [])
                entry["parents"] = list(dict.fromkeys(entry_parents + parents))
            if entry.get("redundant") is None:
                try:
                    entry["redundant"] = bool(analyze_redundancy(path))
                except Exception:  # pragma: no cover - best effort
                    self.logger.exception("redundancy analysis failed for %s", path)
                    entry["redundant"] = False

            if rel in seen:
                continue
            seen.add(rel)

            if not path.exists():
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
                        name = alias.name
                        mod_path = repo / Path(*name.split(".")).with_suffix(".py")
                        pkg_init = repo / Path(*name.split(".")) / "__init__.py"
                        dep = (
                            mod_path
                            if mod_path.exists()
                            else pkg_init
                            if pkg_init.exists()
                            else None
                        )
                        if dep is not None:
                            dep_rel = dep.relative_to(repo).as_posix()
                            dep_parents = [rel] + parents
                            dep_entry = self.orphan_traces.setdefault(
                                dep_rel, {"parents": [], "redundant": None}
                            )
                            dep_entry["parents"] = list(
                                dict.fromkeys(dep_entry.get("parents", []) + dep_parents)
                            )
                            queue.append((dep, dep_parents))
                elif isinstance(node, ast.ImportFrom):
                    if node.level:
                        if node.level - 1 <= len(pkg_parts):
                            base_prefix = pkg_parts[: len(pkg_parts) - node.level + 1]
                        else:
                            base_prefix = []
                    else:
                        base_prefix = pkg_parts
                    if node.module:
                        parts = base_prefix + node.module.split(".")
                        mod_path = repo / Path(*parts).with_suffix(".py")
                        pkg_init = repo / Path(*parts) / "__init__.py"
                        dep = (
                            mod_path
                            if mod_path.exists()
                            else pkg_init
                            if pkg_init.exists()
                            else None
                        )
                        if dep is not None:
                            dep_rel = dep.relative_to(repo).as_posix()
                            dep_parents = [rel] + parents
                            dep_entry = self.orphan_traces.setdefault(
                                dep_rel, {"parents": [], "redundant": None}
                            )
                            dep_entry["parents"] = list(
                                dict.fromkeys(dep_entry.get("parents", []) + dep_parents)
                            )
                            queue.append((dep, dep_parents))
                        for alias in node.names:
                            if alias.name == "*":
                                continue
                            sub_parts = parts + alias.name.split(".")
                            mod_path = repo / Path(*sub_parts).with_suffix(".py")
                            pkg_init = repo / Path(*sub_parts) / "__init__.py"
                            dep = (
                                mod_path
                                if mod_path.exists()
                                else pkg_init
                                if pkg_init.exists()
                                else None
                            )
                            if dep is not None:
                                dep_rel = dep.relative_to(repo).as_posix()
                                dep_parents = [rel] + parents
                                dep_entry = self.orphan_traces.setdefault(
                                    dep_rel, {"parents": [], "redundant": None}
                                )
                                dep_entry["parents"] = list(
                                    dict.fromkeys(
                                        dep_entry.get("parents", []) + dep_parents
                                    )
                                )
                                queue.append((dep, dep_parents))
                    elif node.names:
                        for alias in node.names:
                            name = ".".join(base_prefix + alias.name.split("."))
                            mod_path = repo / Path(*name.split(".")).with_suffix(".py")
                            pkg_init = repo / Path(*name.split(".")) / "__init__.py"
                            dep = (
                                mod_path
                                if mod_path.exists()
                                else pkg_init
                                if pkg_init.exists()
                                else None
                            )
                            if dep is not None:
                                dep_rel = dep.relative_to(repo).as_posix()
                                dep_parents = [rel] + parents
                                dep_entry = self.orphan_traces.setdefault(
                                    dep_rel, {"parents": [], "redundant": None}
                                )
                                dep_entry["parents"] = list(
                                    dict.fromkeys(
                                        dep_entry.get("parents", []) + dep_parents
                                    )
                                )
                                queue.append((dep, dep_parents))

        return seen

    # ------------------------------------------------------------------
    def _discover_orphans(self, recursive: bool | None = None) -> list[str]:
        """Return orphan modules detected in the repository.

        This helper performs discovery only; persisting the results is handled
        by :meth:`_run_once` after combining orphan and isolated modules. The
        ``recursive`` behaviour can be controlled via the
        ``SELF_TEST_RECURSIVE_ORPHANS`` environment variable which, when set to
        a truthy value, enables recursive dependency discovery regardless of the
        instance configuration.
        """
        if recursive is None:
            env_val = os.getenv("SELF_TEST_RECURSIVE_ORPHANS")
            if env_val is None:
                env_val = os.getenv("SANDBOX_RECURSIVE_ORPHANS")
            if env_val is not None:
                recursive = env_val.lower() in ("1", "true", "yes")
            else:
                recursive = self.recursive_orphans

        modules: list[str]
        if recursive:
            from sandbox_runner import discover_recursive_orphans as _discover

            trace = _discover(
                str(Path.cwd()),
                module_map=str(Path("sandbox_data") / "module_map.json"),
            )
            self.orphan_traces = {}
            for k, v in trace.items():
                info: dict[str, Any] = {
                    "parents": [
                        str(Path(*p.split(".")).with_suffix(".py"))
                        for p in (
                            v.get("parents") if isinstance(v, dict) else v
                        )
                    ]
                }
                if isinstance(v, dict) and "redundant" in v:
                    info["redundant"] = bool(v["redundant"])
                self.orphan_traces[str(Path(*k.split(".")).with_suffix(".py"))] = info
            modules = list(self.orphan_traces.keys())
            modules = [str(Path(m)) for m in sorted(self._collect_recursive(modules))]
        else:
            from scripts.find_orphan_modules import find_orphan_modules

            repo = Path(os.getenv("SANDBOX_REPO_PATH", ".")).resolve()
            modules = []
            for p in find_orphan_modules(Path.cwd()):
                try:
                    rel = Path(p).resolve().relative_to(repo).as_posix()
                except Exception:
                    rel = str(p)
                modules.append(rel)
            self.orphan_traces.update({m: {"parents": []} for m in modules})

        filtered: list[str] = []
        for m in modules:
            p = Path(m)
            info = self.orphan_traces.setdefault(m, {"parents": []})
            redundant_flag = info.get("redundant")
            if redundant_flag is None:
                try:
                    redundant_flag = bool(analyze_redundancy(p))
                except Exception as exc:  # pragma: no cover - best effort
                    self.logger.exception(
                        "redundancy analysis failed for %s: %s", p, exc
                    )
                    redundant_flag = False
                info["redundant"] = redundant_flag
            if redundant_flag:
                self.logger.info(
                    "redundant module skipped", extra={"module": m}
                )
                continue
            filtered.append(m)

        return filtered

    # ------------------------------------------------------------------
    def _discover_isolated(self, recursive: bool | None = None) -> list[str]:
        """Return modules discovered by ``discover_isolated_modules``.

        Persistence of the results is handled by the caller.  This wrapper
        records basic metadata for the modules returned by the helper.
        """
        from scripts.discover_isolated_modules import discover_isolated_modules

        modules = discover_isolated_modules(
            Path.cwd(),
            recursive=self.recursive_isolated if recursive is None else recursive,
        )

        roots: list[str] = []
        for m in modules:
            key = str(Path(m))
            entry = self.orphan_traces.setdefault(key, {"parents": []})
            # ``discover_isolated_modules`` only returns non-redundant modules but
            # we record the classification explicitly for later reference.
            entry["redundant"] = False
            roots.append(key)

        # Follow dependency chains so helper modules are recorded alongside the
        # isolated roots. ``_collect_recursive`` updates ``orphan_traces`` with
        # parent relationships and redundancy flags for all discovered modules.
        return [str(Path(p)) for p in sorted(self._collect_recursive(roots))]

    # ------------------------------------------------------------------
    def _clean_orphan_list(self, modules: Iterable[str]) -> None:
        """Prune ``sandbox_data/orphan_modules.json`` after integration.

        ``modules`` should contain the root modules that were successfully
        merged.  Any helper imports recorded in ``self.orphan_traces`` for those
        modules are also removed from the orphan list.  Entries marked
        ``{"redundant": true}`` are preserved for later auditing.
        """

        path = Path("sandbox_data") / "orphan_modules.json"
        if not path.exists():
            return

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
        to_remove: set[str] = set(roots)
        for mod, info in self.orphan_traces.items():
            parents = [_norm(p) for p in info.get("parents", [])]
            if mod in roots or any(parent in roots for parent in parents):
                to_remove.add(mod)

        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh) or []
        except Exception:
            self.logger.exception("failed to load orphan modules")
            return

        changed = False
        if isinstance(data, list):
            remaining: list[str] = []
            for m in data:
                norm = _norm(m)
                if norm in to_remove and not self.orphan_traces.get(norm, {}).get("redundant"):
                    changed = True
                    continue
                remaining.append(m)
            if changed:
                path.write_text(json.dumps(sorted(set(remaining)), indent=2))
        elif isinstance(data, dict):
            remaining_dict: dict[str, Any] = {}
            for k, info in data.items():
                norm = _norm(k)
                if norm in to_remove:
                    redundant = None
                    if isinstance(info, dict):
                        redundant = info.get("redundant")
                    redundant = self.orphan_traces.get(norm, {}).get(
                        "redundant", redundant
                    )
                    if redundant:
                        remaining_dict[k] = {"redundant": True}
                    else:
                        changed = True
                        continue
                else:
                    remaining_dict[k] = info
            if changed:
                path.write_text(json.dumps(remaining_dict, indent=2, sort_keys=True))
        else:  # pragma: no cover - unexpected structure
            self.logger.error("invalid orphan module cache format: %s", type(data))

    # ------------------------------------------------------------------
    async def _run_once(self, *, refresh_orphans: bool = False) -> None:
        other_args = [a for a in self.pytest_args if a.startswith("-")]
        paths = [a for a in self.pytest_args if not a.startswith("-")]
        if not paths:
            paths = [None]

        path = Path("sandbox_data") / "orphan_modules.json"
        existing: list[str] = []
        if path.exists() and not refresh_orphans:
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh) or []
                    if isinstance(data, list):
                        existing = [str(p) for p in data]
            except Exception:
                self.logger.exception("failed to load orphan modules")

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

        if self.discover_orphans:
            try:
                found = self._discover_orphans()
                discovered.extend(found)
                orphan_list.extend(found)
            except Exception:
                self.logger.exception("failed to auto-discover orphan modules")

        if self.discover_isolated:
            try:
                found = self._discover_isolated()
                discovered.extend(found)
                orphan_list.extend(found)
            except Exception:
                self.logger.exception("failed to auto-discover isolated modules")

        if orphan_list:
            orphan_list = list(dict.fromkeys(orphan_list))
        redundant_list = [
            m for m, info in self.orphan_traces.items() if info.get("redundant")
        ]
        if orphan_list:
            orphan_list = [m for m in orphan_list if m not in redundant_list]

        combined_file = list(dict.fromkeys(existing + discovered))
        if combined_file or path.exists():
            try:
                path.parent.mkdir(exist_ok=True)
                path.write_text(json.dumps(combined_file, indent=2))
                new_modules = [m for m in combined_file if m not in existing]
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

        # Always include newly discovered modules in the test run.
        for m in orphan_list:
            if m not in paths:
                paths.append(m)

        self._state = None

        all_orphans = set(orphan_list) | set(redundant_list)
        orphan_set = set(orphan_list)

        queue: list[str] = [
            p or ""
            for p in paths
            if not (p and str(Path(p)) in redundant_list)
        ]
        self._save_state(queue, passed, failed, coverage_total, runtime_total)
        proc_info: list[tuple[list[str], str | None, bool, str | None, str]] = []

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
            ) -> tuple[int, int, float, float, bool, str, str, str]:
                report: dict[str, Any] = {}
                out: bytes = b""
                err: bytes = b""
                attempts = self.container_retries + 1 if is_container else 1
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
                    except asyncio.TimeoutError:
                        proc.kill()
                        await proc.wait()
                        if is_container and name:
                            await self._force_remove_container(name)
                        try:
                            self_test_container_timeouts_total.inc()
                        except Exception:
                            self.logger.exception("failed to update container timeout metric")
                        if attempt == attempts - 1:
                            self.logger.error("self test container timed out")
                            break
                        continue

                    if proc.returncode != 0 and is_container and attempt < attempts - 1:
                        if name:
                            await self._force_remove_container(name)
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
                return pcount, fcount, cov, runtime, failed_flag, out_snip, err_snip, log_snip

            tasks = [asyncio.create_task(_process(cmd, tmp, is_c, name)) for cmd, tmp, is_c, name, _ in proc_info]

            results = await asyncio.gather(*tasks, return_exceptions=True)
            first_exc: Exception | None = None
            any_failed = False
            stdout_snip = ""
            stderr_snip = ""
            logs_snip = ""
            orphan_failed: list[str] = []
            orphan_passed: list[str] = []
            passed_set: list[str] = []

            for (cmd, tmp, is_c, name, p), res in zip(proc_info, results):
                if isinstance(res, Exception):
                    if first_exc is None:
                        first_exc = res
                    continue
                pcount, fcount, cov, runtime, failed_flag, out_snip, err_snip, log_snip = res
                any_failed = any_failed or failed_flag
                passed += pcount
                failed += fcount
                coverage_total += cov
                runtime_total += runtime
                if failed_flag:
                    stdout_snip += out_snip
                    stderr_snip += err_snip
                    logs_snip += log_snip
                    if p in orphan_set:
                        orphan_failed.append(p)

                if p in queue:
                    queue.remove(p)
                if p in orphan_set and not failed_flag:
                    orphan_passed.append(p)
                self._save_state(queue, passed, failed, coverage_total, runtime_total)

                if self.result_callback:
                    partial = {
                        "passed": passed,
                        "failed": failed,
                        "coverage": coverage_total / max(len(paths), 1),
                        "runtime": runtime_total,
                    }
                    try:
                        self.result_callback(partial)
                    except Exception:
                        self.logger.exception("result callback failed")

            try:
                self.error_logger.db.add_test_result(passed, failed)
            except Exception:  # pragma: no cover - best effort
                self.logger.exception("failed to store test results")

            coverage = coverage_total / max(len(proc_info), 1)
            runtime = runtime_total
            self.results = {
                "passed": passed,
                "failed": failed,
                "coverage": coverage,
                "runtime": runtime,
            }
            passed_set: list[str] = []
            if all_orphans:
                passed_set = [p for p in orphan_passed if p not in orphan_failed]
                redundant_set = sorted(set(redundant_list))
                self.results["orphan_total"] = len(all_orphans)
                self.results["orphan_failed"] = len(orphan_failed)
                self.results["orphan_passed"] = sorted(passed_set)
                self.results["orphan_redundant"] = redundant_set
                if orphan_failed:
                    self.results["orphan_failed_modules"] = sorted(orphan_failed)
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

            if passed_set:
                integrate_mods = [
                    m for m in passed_set if not self.orphan_traces.get(m, {}).get("redundant")
                ]
                cleaned = False
                if self.integration_callback and integrate_mods:
                    try:
                        self.integration_callback(integrate_mods)
                    except Exception:
                        self.logger.exception("orphan integration failed")
                    else:
                        cleaned = self.integration_callback is self._default_integration
                if self.clean_orphans and integrate_mods and not cleaned:
                    try:
                        self._clean_orphan_list(integrate_mods)
                    except Exception:
                        self.logger.exception("failed to clean orphan list")

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

    async def _schedule_loop(self, interval: float, *, refresh_orphans: bool = False) -> None:
        assert self._async_stop is not None
        while not self._async_stop.is_set():
            try:
                await self._run_once(refresh_orphans=refresh_orphans)
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
    def run_once(self, *, refresh_orphans: bool = False) -> dict[str, Any]:
        """Execute the self tests once and return the results.

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

        return self.results or {}

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

    run = sub.add_parser("run", help="Run self tests once")
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
        "--include-orphans",
        action="store_true",
        help="Also test modules listed in sandbox_data/orphan_modules.json",
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
        help="Automatically run discover_isolated_modules and append results",
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
        help="Recursively discover dependent orphan chains (alias: --recursive-orphans)",
    )
    run.add_argument(
        "--no-recursive-include",
        "--no-recursive-orphans",
        dest="recursive_orphans",
        action="store_false",
        help="Do not recurse through orphan dependencies",
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
        "--include-orphans",
        action="store_true",
        help="Also test modules listed in sandbox_data/orphan_modules.json",
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
        "--no-container",
        dest="use_container",
        action="store_false",
        help="Run tests on the host instead of a container",
    )
    sched.set_defaults(use_container=True)

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

    val = "1" if args.recursive_orphans else "0"
    os.environ["SANDBOX_RECURSIVE_ORPHANS"] = val
    os.environ["SELF_TEST_RECURSIVE_ORPHANS"] = val
    recursive_orphans = args.recursive_orphans
    set_recursive_isolated(args.recursive_isolated)

    os.environ["SANDBOX_AUTO_INCLUDE_ISOLATED"] = "1" if args.discover_isolated else "0"
    os.environ["SELF_TEST_AUTO_INCLUDE_ISOLATED"] = "1" if args.discover_isolated else "0"

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
            clean_orphans=args.clean_orphans,
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
            clean_orphans=args.clean_orphans,
        )
        try:
            service.run_scheduled(interval=args.interval, refresh_orphans=args.refresh_orphans)
        except KeyboardInterrupt:
            pass
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
