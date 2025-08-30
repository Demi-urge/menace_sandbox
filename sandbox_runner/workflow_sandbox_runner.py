"""Isolated workflow execution with telemetry support.

This module provides :class:`WorkflowSandboxRunner` which runs one or more
callables inside a temporary directory or an OS-level sandbox.  File system
access is redirected to that directory and, when ``safe_mode`` is enabled,
common networking libraries are patched so requests either raise
:class:`RuntimeError` or invoke supplied mock handlers.  File writes are
confined to the sandbox and can also be redirected to custom handlers such as
in-memory buffers.  Access to kernel introspection paths such as ``/proc`` is
left untouched so runners may read process information.  Each executed callable
has execution time, memory usage, CPU time and errors recorded and aggregated
into a :class:`RunMetrics` instance.  When ``use_subprocess`` is enabled the
workflow executes in a separate process with CPU and memory quotas enforced via
``resource`` limits or cgroups.  Optional audit hooks receive callbacks for
file and network access attempts, allowing external logging of sandboxed
behaviour including blocked network traffic.

Safe mode guarantees
--------------------

When ``safe_mode`` is ``True`` the runner enforces two security properties:

* Outbound network access via :mod:`socket`, :mod:`urllib`, :mod:`requests`
  and :mod:`httpx` is blocked.  Tests that expect network traffic must supply
  a response through ``test_data`` or ``network_mocks``.
* File writes are disallowed and path resolution prevents escaping the
  sandbox.  To permit writes or to intercept them, callers may provide
  handlers through ``fs_mocks``.

These guarantees ensure workflows cannot reach external systems or modify the
host filesystem unless explicitly allowed by the test harness.
"""

from __future__ import annotations

import builtins
import contextlib
import inspect
import asyncio
import json
import os
import pathlib
import shutil
import tempfile
import urllib.parse
import signal
import threading
import _thread
import logging
import multiprocessing
import traceback
from dataclasses import dataclass, field
from time import perf_counter, process_time
from typing import Any, Callable, Iterable, Mapping
from unittest import mock

try:  # pragma: no cover - resource module may be missing on some platforms
    import resource  # type: ignore
except Exception:  # pragma: no cover - resource unavailable
    resource = None  # type: ignore

try:  # pragma: no cover - psutil is optional
    import psutil  # type: ignore
except Exception:  # pragma: no cover - psutil missing
    psutil = None  # type: ignore

try:  # pragma: no cover - optional metrics exporter
    from metrics_exporter import Gauge as _Gauge  # type: ignore
except Exception:  # pragma: no cover - metrics exporter missing
    _Gauge = None  # type: ignore

try:  # pragma: no cover - optional meta logger
    from .meta_logger import _SandboxMetaLogger  # type: ignore
except Exception:  # pragma: no cover - meta logger missing
    _SandboxMetaLogger = None  # type: ignore

import tracemalloc

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
@dataclass
class ModuleMetrics:
    """Telemetry captured for a single executed module."""

    name: str
    duration: float
    cpu_before: float
    cpu_after: float
    cpu_delta: float
    memory_before: int
    memory_after: int
    memory_delta: int
    memory_peak: int
    success: bool
    exception: str | None = None
    result: Any | None = None
    fixtures: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class RunMetrics:
    """Aggregated metrics across all executed modules."""

    modules: list[ModuleMetrics] = field(default_factory=list)
    crash_count: int = 0


class EmptyWorkflowError(RuntimeError):
    """Raised when a workflow contains no actionable steps."""

    pass


# ---------------------------------------------------------------------------
class WorkflowSandboxRunner:
    """Execute workflows inside an isolated sandbox."""

    def __init__(self) -> None:
        # ``metrics`` retains the raw :class:`RunMetrics` instance while
        # ``telemetry`` exposes a serialisable summary suitable for emission to
        # loggers or exporters.
        self.metrics: RunMetrics | None = None
        self.telemetry: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    def _resolve(self, root: pathlib.Path, path: str | pathlib.Path) -> pathlib.Path:
        """Return a normalised path within ``root``.

        ``path`` may be relative or absolute.  Absolute paths already located
        inside ``root`` are preserved while all other paths are treated as being
        relative to ``root``.  The final resolved path is verified to reside
        inside ``root`` to guard against ``..`` segments or symlink escapes.  A
        :class:`RuntimeError` is raised if the normalised path would point
        outside the sandbox.
        """

        root = root.resolve()
        p = pathlib.Path(path)
        if p.is_absolute():
            try:
                p = p.relative_to(root)
            except ValueError:
                p = pathlib.Path(*p.parts[1:])
        candidate = (root / p).resolve()
        if not candidate.is_relative_to(root):
            raise RuntimeError("path escapes sandbox")
        return candidate

    # ------------------------------------------------------------------
    def run(
        self,
        workflow: Callable[[], Any] | Iterable[Callable[[], Any]],
        *,
        safe_mode: bool = False,
        test_data: Mapping[str, str | bytes] | None = None,
        network_mocks: Mapping[str, Callable[..., Any]] | None = None,
        fs_mocks: Mapping[str, Callable[..., Any]] | None = None,
        module_fixtures: Mapping[str, Mapping[str, Any]] | None = None,
        roi_delta: float | None = None,
        timeout: float | None = None,
        memory_limit: int | None = None,
        cpu_limit: int | None = None,
        use_subprocess: bool = False,
        audit_hook: Callable[[str, Mapping[str, Any]], None] | None = None,
    ) -> RunMetrics:
        """Execute ``workflow`` inside a sandbox and return telemetry.

        ``workflow`` may be a single callable or an iterable of callables.
        ``test_data`` provides stubbed file contents or network responses.  Keys
        containing an ``http`` or ``https`` scheme are treated as URLs and
        returned via monkeypatched networking libraries.  All other keys are
        assumed to represent file paths and are written into the sandbox before
        execution.

        ``network_mocks`` and ``fs_mocks`` allow callers to supply custom
        functions for the patched network and filesystem helpers respectively.
        ``network_mocks`` must map full URLs or URL prefixes to callables.  Any
        network request performed while ``safe_mode`` is enabled that does not
        match an entry in ``network_mocks`` (or ``test_data``) will raise a
        :class:`RuntimeError`.

        ``module_fixtures`` maps module names to fixture dictionaries.  Each
        fixture may contain ``"files"`` and ``"env"`` mappings that are applied
        before the module executes.  File fixtures are written into the sandbox
        and environment variables are temporarily set for the duration of the
        module's execution and restored afterwards.

        ``timeout`` specifies the maximum number of seconds each module is
        allowed to run.  Exceeding the limit aborts the module and records a
        crash.

        ``memory_limit`` sets an upper bound on RSS memory usage for the
        running process in bytes.  When exceeded the module is interrupted and
        the event recorded as a crash.  This requires ``psutil`` to be
        available; otherwise the limit is ignored.

        ``cpu_limit`` specifies the maximum amount of CPU time (in seconds)
        available to the workflow when executed in a subprocess.  This is
        enforced using ``resource`` limits or cgroups when available.

        When ``use_subprocess`` is ``True`` the entire workflow executes in a
        separate Python process with the supplied CPU and memory limits applied.
        This provides an additional OS-level isolation layer.

        ``audit_hook`` if provided is invoked for every file and network access
        attempt.  The first argument is a string describing the event and the
        second a mapping containing event metadata.  Hooks must be picklable if
        ``use_subprocess`` is enabled.
        """

        test_data = dict(test_data or {})
        network_mocks = dict(network_mocks or {})
        fs_mocks = dict(fs_mocks or {})
        module_fixtures = {k: dict(v) for k, v in (module_fixtures or {}).items()}

        file_data: dict[str, str | bytes] = {}
        network_data: dict[str, str | bytes] = {}
        for key, value in test_data.items():
            scheme = urllib.parse.urlparse(key).scheme
            if scheme in {"http", "https"}:
                network_data[key] = value
            else:
                file_data[key] = value

        if use_subprocess:
            parent_conn, child_conn = multiprocessing.Pipe()
            params = {
                "safe_mode": safe_mode,
                "test_data": test_data,
                "network_mocks": network_mocks,
                "fs_mocks": fs_mocks,
                "module_fixtures": module_fixtures,
                "roi_delta": roi_delta,
                "timeout": timeout,
                "memory_limit": memory_limit,
                "cpu_limit": cpu_limit,
                "audit_hook": audit_hook,
                "use_subprocess": False,
            }

            p = multiprocessing.Process(
                target=_subprocess_worker, args=(child_conn, workflow, params)
            )
            p.start()
            p.join()
            if p.is_alive():
                p.kill()
                p.join()
            metrics, telemetry = parent_conn.recv()
            self.metrics = metrics
            self.telemetry = telemetry
            return metrics

        if timeout is None or memory_limit is None:
            try:  # pragma: no cover - optional settings
                from sandbox_settings import SandboxSettings

                _settings = SandboxSettings()
            except Exception:  # pragma: no cover - settings missing
                _settings = None
            if timeout is None and _settings and _settings.default_module_timeout is not None:
                timeout = _settings.default_module_timeout
            if memory_limit is None and _settings and _settings.default_memory_limit is not None:
                memory_limit = _settings.default_memory_limit

        proc = psutil.Process() if psutil else None

        try:
            from .environment import _patched_imports
        except Exception:  # pragma: no cover - fallback when environment unavailable
            @contextlib.contextmanager
            def _patched_imports():
                yield

        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
            root = pathlib.Path(tmp).resolve()
            stack.enter_context(_patched_imports())

            def _audit(event: str, **info: Any) -> None:
                if audit_hook:
                    try:
                        audit_hook(event, info)
                    except Exception:
                        logger.exception("audit hook error")

            funcs = [workflow] if callable(workflow) else list(workflow)
            if not funcs:
                raise EmptyWorkflowError("workflow contains no actionable steps")

            # Copy each module's source file into the sandbox for completeness.
            for fn in funcs:
                try:
                    src = inspect.getsourcefile(fn) or inspect.getfile(fn)
                    if src:
                        shutil.copy2(src, root / pathlib.Path(src).name)
                except Exception as exc:  # pragma: no cover - best effort
                    logger.warning("failed to copy source for %s: %s", fn, exc, exc_info=True)

            # ------------------------------------------------------------------
            # Redirect tempfile helpers to operate inside the sandbox
            original_tempdir = tempfile.tempdir
            tempfile.tempdir = str(root)
            stack.callback(lambda: setattr(tempfile, "tempdir", original_tempdir))

            original_named_temporary_file = tempfile.NamedTemporaryFile
            original_temporary_file = tempfile.TemporaryFile
            original_spooled_temporary_file = tempfile.SpooledTemporaryFile

            def sandbox_named_temporary_file(*a, dir=None, **kw):
                target_dir = self._resolve(root, dir or root)
                fn = fs_mocks.get("tempfile.NamedTemporaryFile")
                if fn:
                    return fn(*a, dir=str(target_dir), **kw)
                if safe_mode:
                    raise RuntimeError("file write disabled in safe_mode")
                return original_named_temporary_file(*a, dir=str(target_dir), **kw)

            def sandbox_temporary_file(*a, dir=None, **kw):
                target_dir = self._resolve(root, dir or root)
                fn = fs_mocks.get("tempfile.TemporaryFile")
                if fn:
                    return fn(*a, dir=str(target_dir), **kw)
                if safe_mode:
                    raise RuntimeError("file write disabled in safe_mode")
                return original_temporary_file(*a, dir=str(target_dir), **kw)

            def sandbox_spooled_temporary_file(*a, dir=None, **kw):
                target_dir = self._resolve(root, dir or root)
                fn = fs_mocks.get("tempfile.SpooledTemporaryFile")
                if fn:
                    return fn(*a, dir=str(target_dir), **kw)
                if safe_mode:
                    raise RuntimeError("file write disabled in safe_mode")
                return original_spooled_temporary_file(*a, dir=str(target_dir), **kw)

            stack.enter_context(
                mock.patch.object(tempfile, "NamedTemporaryFile", sandbox_named_temporary_file)
            )
            stack.enter_context(
                mock.patch.object(tempfile, "TemporaryFile", sandbox_temporary_file)
            )
            stack.enter_context(
                mock.patch.object(
                    tempfile, "SpooledTemporaryFile", sandbox_spooled_temporary_file
                )
            )

            # ------------------------------------------------------------------
            # Monkeypatch filesystem helpers so all paths resolve inside ``root``
            original_open = builtins.open

            def sandbox_open(file, mode="r", *a, **kw):
                file_path = os.fspath(file)
                if file_path.startswith("/proc/"):
                    return original_open(file, mode, *a, **kw)
                p = pathlib.Path(file_path)
                path = self._resolve(root, file_path)
                if not path.is_relative_to(root):
                    raise RuntimeError("path escapes sandbox")
                if any(m in mode for m in ("w", "a", "x", "+")):
                    fn = fs_mocks.get("open")
                    if fn:
                        return fn(path, mode, *a, **kw)
                    if safe_mode and (not p.is_absolute() or p.is_relative_to(root)):
                        raise RuntimeError("file write disabled in safe_mode")
                    path.parent.mkdir(parents=True, exist_ok=True)
                _audit("file_open", path=str(path), mode=mode)
                return original_open(path, mode, *a, **kw)

            stack.enter_context(mock.patch("builtins.open", sandbox_open))

            original_path_open = pathlib.Path.open
            original_write_text = pathlib.Path.write_text
            original_read_text = pathlib.Path.read_text
            original_write_bytes = pathlib.Path.write_bytes
            original_read_bytes = pathlib.Path.read_bytes
            original_path_mkdir = pathlib.Path.mkdir
            original_path_unlink = pathlib.Path.unlink
            original_path_rmdir = pathlib.Path.rmdir

            def path_open(path_obj, *a, **kw):
                mode = a[0] if a else kw.get("mode", "r")
                raw = os.fspath(path_obj)
                if raw.startswith("/proc/"):
                    return original_path_open(path_obj, *a, **kw)
                p = pathlib.Path(raw)
                path = self._resolve(root, raw)
                if any(m in mode for m in ("w", "a", "x", "+")):
                    fn = fs_mocks.get("pathlib.Path.open")
                    if fn:
                        return fn(path, *a, **kw)
                    if safe_mode and (not p.is_absolute() or p.is_relative_to(root)):
                        raise RuntimeError("file write disabled in safe_mode")
                    path.parent.mkdir(parents=True, exist_ok=True)
                _audit("file_open", path=str(path), mode=mode)
                return original_path_open(path, *a, **kw)

            def path_write_text(path_obj, data, *a, **kw):
                raw = os.fspath(path_obj)
                if raw.startswith("/proc/"):
                    return original_write_text(path_obj, data, *a, **kw)
                p = pathlib.Path(raw)
                path = self._resolve(root, raw)
                if not path.is_relative_to(root):
                    raise RuntimeError("path escapes sandbox")
                fn = fs_mocks.get("pathlib.Path.write_text")
                if fn:
                    return fn(path, data, *a, **kw)
                if safe_mode and (not p.is_absolute() or p.is_relative_to(root)):
                    raise RuntimeError("file write disabled in safe_mode")
                path.parent.mkdir(parents=True, exist_ok=True)
                _audit("file_write", path=str(path))
                return original_write_text(path, data, *a, **kw)

            def path_read_text(path_obj, *a, **kw):
                raw = os.fspath(path_obj)
                if raw.startswith("/proc/"):
                    return original_read_text(path_obj, *a, **kw)
                path = self._resolve(root, raw)
                _audit("file_read", path=str(path))
                return original_read_text(path, *a, **kw)

            def path_write_bytes(path_obj, data, *a, **kw):
                raw = os.fspath(path_obj)
                if raw.startswith("/proc/"):
                    return original_write_bytes(path_obj, data, *a, **kw)
                p = pathlib.Path(raw)
                path = self._resolve(root, raw)
                if not path.is_relative_to(root):
                    raise RuntimeError("path escapes sandbox")
                fn = fs_mocks.get("pathlib.Path.write_bytes")
                if fn:
                    return fn(path, data, *a, **kw)
                if safe_mode and (not p.is_absolute() or p.is_relative_to(root)):
                    raise RuntimeError("file write disabled in safe_mode")
                path.parent.mkdir(parents=True, exist_ok=True)
                _audit("file_write", path=str(path))
                return original_write_bytes(path, data, *a, **kw)

            def path_read_bytes(path_obj, *a, **kw):
                raw = os.fspath(path_obj)
                if raw.startswith("/proc/"):
                    return original_read_bytes(path_obj, *a, **kw)
                path = self._resolve(root, raw)
                _audit("file_read", path=str(path))
                return original_read_bytes(path, *a, **kw)

            def path_mkdir(path_obj, *a, **kw):
                raw = os.fspath(path_obj)
                if raw.startswith("/proc/"):
                    return original_path_mkdir(path_obj, *a, **kw)
                p = pathlib.Path(raw)
                path = self._resolve(root, raw)
                fn = fs_mocks.get("pathlib.Path.mkdir")
                if fn:
                    return fn(path, *a, **kw)
                if safe_mode and (not p.is_absolute() or p.is_relative_to(root)):
                    raise RuntimeError("file write disabled in safe_mode")
                _audit("file_mkdir", path=str(path))
                return original_path_mkdir(path, *a, **kw)

            def path_unlink(path_obj, *a, **kw):
                raw = os.fspath(path_obj)
                if raw.startswith("/proc/"):
                    return original_path_unlink(path_obj, *a, **kw)
                p = pathlib.Path(raw)
                path = self._resolve(root, raw)
                fn = fs_mocks.get("pathlib.Path.unlink")
                if fn:
                    return fn(path, *a, **kw)
                if safe_mode and (not p.is_absolute() or p.is_relative_to(root)):
                    raise RuntimeError("file write disabled in safe_mode")
                _audit("file_unlink", path=str(path))
                return original_path_unlink(path, *a, **kw)

            def path_rmdir(path_obj, *a, **kw):
                raw = os.fspath(path_obj)
                if raw.startswith("/proc/"):
                    return original_path_rmdir(path_obj, *a, **kw)
                p = pathlib.Path(raw)
                path = self._resolve(root, raw)
                fn = fs_mocks.get("pathlib.Path.rmdir")
                if fn:
                    return fn(path, *a, **kw)
                if safe_mode and (not p.is_absolute() or p.is_relative_to(root)):
                    raise RuntimeError("file write disabled in safe_mode")
                _audit("file_rmdir", path=str(path))
                return original_path_rmdir(path, *a, **kw)

            stack.enter_context(mock.patch.object(pathlib.Path, "open", path_open))
            stack.enter_context(
                mock.patch.object(pathlib.Path, "write_text", path_write_text)
            )
            stack.enter_context(
                mock.patch.object(pathlib.Path, "read_text", path_read_text)
            )
            stack.enter_context(
                mock.patch.object(pathlib.Path, "write_bytes", path_write_bytes)
            )
            stack.enter_context(
                mock.patch.object(pathlib.Path, "read_bytes", path_read_bytes)
            )
            stack.enter_context(
                mock.patch.object(pathlib.Path, "mkdir", path_mkdir)
            )
            stack.enter_context(
                mock.patch.object(pathlib.Path, "unlink", path_unlink)
            )
            stack.enter_context(
                mock.patch.object(pathlib.Path, "rmdir", path_rmdir)
            )

            original_mkdir = os.mkdir
            original_makedirs = os.makedirs
            original_rmdir = os.rmdir
            original_removedirs = os.removedirs
            original_os_open = os.open
            original_stat = os.stat
            original_rmtree = shutil.rmtree
            original_remove = os.remove
            original_unlink = os.unlink
            original_rename = os.rename
            original_replace = os.replace
            original_copy = shutil.copy
            original_copy2 = shutil.copy2
            original_copyfile = shutil.copyfile
            original_copytree = shutil.copytree
            original_move = shutil.move

            def sandbox_mkdir(path, mode=0o777, *a, **kw):
                p = pathlib.Path(path)
                s = self._resolve(root, path)
                fn = fs_mocks.get("os.mkdir")
                if fn:
                    return fn(s, mode, *a, **kw)
                if safe_mode and (not p.is_absolute() or p.is_relative_to(root)):
                    raise RuntimeError("file write disabled in safe_mode")
                _audit("file_mkdir", path=str(s))
                return original_mkdir(s, mode, *a, **kw)

            def sandbox_makedirs(path, mode=0o777, exist_ok=False):
                p = pathlib.Path(path)
                s = self._resolve(root, path)
                fn = fs_mocks.get("os.makedirs")
                if fn:
                    return fn(s, mode, exist_ok=exist_ok)
                if safe_mode and (not p.is_absolute() or p.is_relative_to(root)):
                    raise RuntimeError("file write disabled in safe_mode")
                _audit("file_mkdir", path=str(s))
                return original_makedirs(s, mode, exist_ok=exist_ok)

            def sandbox_rmdir(path, *a, **kw):
                raw = os.fspath(path)
                if kw.get("dir_fd") is not None and not os.path.isabs(raw):
                    return original_rmdir(raw, *a, **kw)
                p = pathlib.Path(raw)
                s = self._resolve(root, raw)
                fn = fs_mocks.get("os.rmdir")
                if fn:
                    return fn(s, *a, **kw)
                if safe_mode and (not p.is_absolute() or p.is_relative_to(root)):
                    raise RuntimeError("file write disabled in safe_mode")
                _audit("file_rmdir", path=str(s))
                return original_rmdir(s, *a, **kw)

            def sandbox_removedirs(path, *a, **kw):
                p = pathlib.Path(path)
                s = self._resolve(root, path)
                fn = fs_mocks.get("os.removedirs")
                if fn:
                    return fn(s, *a, **kw)
                if safe_mode and (not p.is_absolute() or p.is_relative_to(root)):
                    raise RuntimeError("file write disabled in safe_mode")
                _audit("file_rmdir", path=str(s))
                return original_removedirs(s, *a, **kw)

            def sandbox_os_open(path, flags, *a, **kw):
                raw = os.fspath(path)
                if raw.startswith("/proc/"):
                    return original_os_open(path, flags, *a, **kw)
                if kw.get("dir_fd") is not None and not os.path.isabs(raw):
                    return original_os_open(path, flags, *a, **kw)
                p = pathlib.Path(raw)
                s = self._resolve(root, raw)
                if flags & (
                    os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
                ):
                    fn = fs_mocks.get("os.open")
                    if fn:
                        return fn(s, flags, *a, **kw)
                    if safe_mode and (not p.is_absolute() or p.is_relative_to(root)):
                        raise RuntimeError("file write disabled in safe_mode")
                    pathlib.Path(s).parent.mkdir(parents=True, exist_ok=True)
                _audit("file_open", path=str(s), flags=flags)
                return original_os_open(s, flags, *a, **kw)

            def sandbox_stat(path, *a, **kw):
                raw = os.fspath(path)
                if raw.startswith("/proc/"):
                    return original_stat(path, *a, **kw)
                if os.path.isabs(raw):
                    try:
                        return original_stat(raw, *a, **kw)
                    except Exception:
                        pass
                current = os.stat
                try:
                    os.stat = original_stat
                    s = self._resolve(root, raw)
                finally:
                    os.stat = current
                fn = fs_mocks.get("os.stat")
                if fn:
                    _audit("file_stat", path=str(s))
                    return fn(s, *a, **kw)
                _audit("file_stat", path=str(s))
                return original_stat(s, *a, **kw)

            def sandbox_rmtree(path, *a, **kw):
                p = pathlib.Path(path)
                s = self._resolve(root, path)
                fn = fs_mocks.get("shutil.rmtree")
                if fn:
                    return fn(s, *a, **kw)
                if safe_mode and (not p.is_absolute() or p.is_relative_to(root)):
                    raise RuntimeError("file write disabled in safe_mode")
                return original_rmtree(s, *a, **kw)

            def sandbox_remove(path, *a, **kw):
                raw = os.fspath(path)
                if kw.get("dir_fd") is not None and not os.path.isabs(raw):
                    return original_remove(raw, *a, **kw)
                p = self._resolve(root, raw)
                fn = fs_mocks.get("os.remove")
                if fn:
                    return fn(p, *a, **kw)
                return original_remove(p, *a, **kw)

            def sandbox_unlink(path, *a, **kw):
                raw = os.fspath(path)
                if kw.get("dir_fd") is not None and not os.path.isabs(raw):
                    return original_unlink(raw, *a, **kw)
                p = self._resolve(root, raw)
                fn = fs_mocks.get("os.unlink")
                if fn:
                    return fn(p, *a, **kw)
                return original_unlink(p, *a, **kw)

            def sandbox_rename(src, dst, *a, **kw):
                s = self._resolve(root, src)
                d = self._resolve(root, dst)
                fn = fs_mocks.get("os.rename")
                if fn:
                    return fn(s, d, *a, **kw)
                pathlib.Path(d).parent.mkdir(parents=True, exist_ok=True)
                return original_rename(s, d, *a, **kw)

            def sandbox_replace(src, dst, *a, **kw):
                s = self._resolve(root, src)
                d = self._resolve(root, dst)
                fn = fs_mocks.get("os.replace")
                if fn:
                    return fn(s, d, *a, **kw)
                pathlib.Path(d).parent.mkdir(parents=True, exist_ok=True)
                return original_replace(s, d, *a, **kw)

            def sandbox_copy(src, dst, *a, **kw):
                s = self._resolve(root, src)
                d = self._resolve(root, dst)
                fn = fs_mocks.get("shutil.copy")
                if fn:
                    return fn(s, d, *a, **kw)
                pathlib.Path(d).parent.mkdir(parents=True, exist_ok=True)
                return original_copy(s, d, *a, **kw)

            def sandbox_copy2(src, dst, *a, **kw):
                s = self._resolve(root, src)
                d = self._resolve(root, dst)
                fn = fs_mocks.get("shutil.copy2")
                if fn:
                    return fn(s, d, *a, **kw)
                pathlib.Path(d).parent.mkdir(parents=True, exist_ok=True)
                return original_copy2(s, d, *a, **kw)

            def sandbox_copyfile(src, dst, *a, **kw):
                s = self._resolve(root, src)
                d = self._resolve(root, dst)
                fn = fs_mocks.get("shutil.copyfile")
                if fn:
                    return fn(s, d, *a, **kw)
                pathlib.Path(d).parent.mkdir(parents=True, exist_ok=True)
                return original_copyfile(s, d, *a, **kw)

            def sandbox_copytree(src, dst, *a, **kw):
                s = self._resolve(root, src)
                d = self._resolve(root, dst)
                fn = fs_mocks.get("shutil.copytree")
                if fn:
                    return fn(s, d, *a, **kw)
                pathlib.Path(d).parent.mkdir(parents=True, exist_ok=True)
                return original_copytree(s, d, *a, **kw)

            def sandbox_move(src, dst, *a, **kw):
                s = self._resolve(root, src)
                d = self._resolve(root, dst)
                fn = fs_mocks.get("shutil.move")
                if fn:
                    return fn(s, d, *a, **kw)
                pathlib.Path(d).parent.mkdir(parents=True, exist_ok=True)
                return original_move(s, d, *a, **kw)

            stack.enter_context(mock.patch("os.mkdir", sandbox_mkdir))
            stack.enter_context(mock.patch("os.makedirs", sandbox_makedirs))
            stack.enter_context(mock.patch("os.rmdir", sandbox_rmdir))
            stack.enter_context(mock.patch("os.removedirs", sandbox_removedirs))
            stack.enter_context(mock.patch("os.open", sandbox_os_open))
            stack.enter_context(mock.patch("os.stat", sandbox_stat))
            stack.enter_context(mock.patch("shutil.rmtree", sandbox_rmtree))
            stack.enter_context(mock.patch("os.remove", sandbox_remove))
            stack.enter_context(mock.patch("os.unlink", sandbox_unlink))
            stack.enter_context(mock.patch("os.rename", sandbox_rename))
            stack.enter_context(mock.patch("os.replace", sandbox_replace))
            stack.enter_context(mock.patch("shutil.copy", sandbox_copy))
            stack.enter_context(mock.patch("shutil.copy2", sandbox_copy2))
            stack.enter_context(mock.patch("shutil.copyfile", sandbox_copyfile))
            stack.enter_context(mock.patch("shutil.copytree", sandbox_copytree))
            stack.enter_context(mock.patch("shutil.move", sandbox_move))

            # Pre-populate any provided file data into the sandbox.
            for name, content in file_data.items():
                real = self._resolve(root, name)
                # ``real.parent`` may require creation even when ``safe_mode`` is
                # enabled. Temporarily restore the original ``os.mkdir`` so that
                # directory creation bypasses sandbox write restrictions.
                current_mkdir = os.mkdir
                try:
                    os.mkdir = original_mkdir
                    original_makedirs(real.parent, exist_ok=True)
                finally:
                    os.mkdir = current_mkdir
                mode = "wb" if isinstance(content, (bytes, bytearray)) else "w"
                with original_open(real, mode) as fh:
                    fh.write(content)

            # ------------------------------------------------------------------
            # Monkeypatch networking primitives

            if safe_mode:
                import socket

                original_socket = socket.socket

                def _blocked_socket(*a, **kw):
                    family = kw.get("family")
                    if family is None and a:
                        family = a[0]
                    if family == socket.AF_UNIX:
                        return original_socket(*a, **kw)
                    _audit("network_blocked", target="socket")
                    raise RuntimeError("network access disabled in safe_mode")

                stack.enter_context(mock.patch("socket.socket", _blocked_socket))

                for _name in [
                    "create_connection",
                    "create_server",
                    "fromfd",
                ]:
                    if hasattr(socket, _name):
                        stack.enter_context(
                            mock.patch(f"socket.{_name}", _blocked_socket)
                        )

            def _response_for(
                content: str | bytes,
                *,
                status: int = 200,
                headers: Mapping[str, str] | None = None,
            ):
                data = content if isinstance(content, (bytes, bytearray)) else str(content).encode()

                class _Resp:
                    def __init__(
                        self,
                        b: bytes,
                        status_code: int,
                        headers: Mapping[str, str] | None = None,
                    ):
                        self.content = b
                        self.status_code = status_code
                        self.headers = dict(headers or {})

                    def read(self) -> bytes:
                        return self.content

                    @property
                    def text(self) -> str:
                        return self.content.decode()

                    def json(self) -> Any:
                        return json.loads(self.text)

                return _Resp(data, status, headers)

            try:  # pragma: no cover - optional dependency
                import requests  # type: ignore

                orig_request = requests.Session.request

                def fake_request(self, method, url, *a, **kw):
                    _audit("network_request", url=url, method=method)
                    if url in network_data:
                        return _response_for(network_data[url])
                    fn = network_mocks.get(url)
                    if fn:
                        return fn(self, method, url, *a, **kw)
                    if safe_mode:
                        _audit("network_blocked", url=url, method=method)
                        raise RuntimeError("network access disabled in safe_mode")
                    return orig_request(self, method, url, *a, **kw)

                stack.enter_context(
                    mock.patch.object(requests.Session, "request", fake_request)
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("requests sandboxing skipped: %s", exc, exc_info=True)

                try:  # pragma: no cover - optional dependency
                    import httpx  # type: ignore

                    def _httpx_response(content: str | bytes) -> httpx.Response:
                        data = (
                            content
                            if isinstance(content, (bytes, bytearray))
                            else str(content).encode()
                        )
                        return httpx.Response(200, content=data)

                    orig_httpx_request = httpx.Client.request

                    def fake_httpx_request(self, method, url, *a, **kw):
                        _audit("network_request", url=url, method=method)
                        if url in network_data:
                            return _httpx_response(network_data[url])
                        fn = network_mocks.get(url)
                        if fn:
                            return fn(self, method, url, *a, **kw)
                        if safe_mode:
                            _audit("network_blocked", url=url, method=method)
                            raise RuntimeError("network access disabled in safe_mode")
                        return orig_httpx_request(self, method, url, *a, **kw)

                    stack.enter_context(
                        mock.patch.object(httpx.Client, "request", fake_httpx_request)
                    )

                    if hasattr(httpx, "AsyncClient"):
                        orig_async_httpx_request = httpx.AsyncClient.request

                        async def fake_async_httpx_request(self, method, url, *a, **kw):
                            _audit("network_request", url=url, method=method)
                            if url in network_data:
                                return _httpx_response(network_data[url])
                            fn = network_mocks.get(url)
                            if fn:
                                res = fn(self, method, url, *a, **kw)
                                if inspect.isawaitable(res):
                                    return await res
                                return res
                            if safe_mode:
                                _audit("network_blocked", url=url, method=method)
                                raise RuntimeError("network access disabled in safe_mode")
                            return await orig_async_httpx_request(self, method, url, *a, **kw)

                        stack.enter_context(
                            mock.patch.object(
                                httpx.AsyncClient, "request", fake_async_httpx_request
                            )
                        )
                except Exception as exc:  # pragma: no cover
                    logger.warning("httpx sandboxing skipped: %s", exc, exc_info=True)

                try:  # pragma: no cover - optional dependency
                    import aiohttp  # type: ignore

                    orig_aio_request = aiohttp.ClientSession._request

                    async def fake_aio_request(self, method, url, *a, **kw):
                        _audit("network_request", url=url, method=method)
                        if url in network_data:
                            data = network_data[url]
                            if isinstance(data, str):
                                data = data.encode()

                            class _AioResp:
                                def __init__(self, b: bytes):
                                    self.status = 200
                                    self._b = b

                                async def read(self) -> bytes:
                                    return self._b

                                async def text(self) -> str:
                                    return self._b.decode()

                            return _AioResp(data)
                        fn = network_mocks.get(url)
                        if fn:
                            res = fn(self, method, url, *a, **kw)
                            if inspect.isawaitable(res):
                                return await res
                            return res
                        if safe_mode:
                            _audit("network_blocked", url=url, method=method)
                            raise RuntimeError("network access disabled in safe_mode")
                        return await orig_aio_request(self, method, url, *a, **kw)

                    stack.enter_context(
                        mock.patch.object(
                            aiohttp.ClientSession, "_request", fake_aio_request
                        )
                    )
                except Exception as exc:  # pragma: no cover
                    logger.warning("aiohttp sandboxing skipped: %s", exc, exc_info=True)

            try:  # pragma: no cover - optional dependency
                import urllib.request as urllib_request  # type: ignore

                orig_urlopen = urllib_request.urlopen

                def fake_urlopen(url, *a, **kw):
                    u = url if isinstance(url, str) else url.get_full_url()
                    _audit("network_request", url=u, method="urlopen")
                    if u in network_data:
                        return _response_for(network_data[u])
                    fn = network_mocks.get(u)
                    if fn:
                        return fn(url, *a, **kw)
                    if safe_mode:
                        _audit("network_blocked", url=u, method="urlopen")
                        raise RuntimeError("network access disabled in safe_mode")
                    return orig_urlopen(url, *a, **kw)

                stack.enter_context(
                    mock.patch.object(urllib_request, "urlopen", fake_urlopen)
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("urllib sandboxing skipped: %s", exc, exc_info=True)

            metrics = RunMetrics()

            for fn in funcs:
                name = getattr(fn, "__name__", repr(fn))
                fixtures = module_fixtures.get(name, {})
                files = dict(fixtures.get("files", {}))
                env_vars = dict(fixtures.get("env", {}))

                # Write per-module file fixtures
                for fname, content in files.items():
                    real = self._resolve(root, fname)
                    real.parent.mkdir(parents=True, exist_ok=True)
                    mode = "wb" if isinstance(content, (bytes, bytearray)) else "w"
                    with original_open(real, mode) as fh:
                        fh.write(content)

                # Temporarily set environment variables for the module
                old_env: dict[str, str | None] = {}
                for key, value in env_vars.items():
                    old_env[key] = os.environ.get(key)
                    # ``os.environ`` expects string values.  Coerce anything
                    # provided via fixtures to ``str`` to avoid ``TypeError``
                    # when callers supply non-string objects such as numbers.
                    os.environ[key] = str(value)

                start = perf_counter()
                cpu_before = cpu_after = 0.0
                mem_before = mem_after = 0
                use_psutil = bool(proc)
                use_psutil_cpu = use_psutil
                if use_psutil_cpu:
                    try:
                        ct = proc.cpu_times()  # type: ignore[union-attr]
                        cpu_before = ct.user + getattr(ct, "system", 0.0)
                    except Exception:
                        use_psutil_cpu = False
                if not use_psutil_cpu:
                    cpu_before = process_time()

                if use_psutil:
                    try:
                        mem_before = proc.memory_info().rss  # type: ignore[union-attr]
                    except Exception:
                        use_psutil = False
                if not use_psutil:
                    tracemalloc.start()
                    mem_before, _ = tracemalloc.get_traced_memory()

                success = True
                error: str | None = None
                result: Any | None = None
                timeout_event = threading.Event()
                mem_event = threading.Event()
                mem_stop = threading.Event()
                timer: threading.Timer | None = None
                mem_thread: threading.Thread | None = None
                old_alarm: Any = None
                old_mem_handler: Any = None

                if timeout:
                    if hasattr(signal, "SIGALRM"):
                        def _timeout_handler(signum, frame):  # pragma: no cover - handler
                            timeout_event.set()
                            raise TimeoutError("module exceeded timeout")

                        old_alarm = signal.signal(signal.SIGALRM, _timeout_handler)
                        signal.setitimer(signal.ITIMER_REAL, timeout)
                    else:  # pragma: no cover - non POSIX
                        timer = threading.Timer(
                            timeout,
                            lambda: (timeout_event.set(), _thread.interrupt_main()),
                        )
                        timer.daemon = True
                        timer.start()

                if memory_limit and proc:
                    def _monitor_mem() -> None:  # pragma: no cover - thread
                        while not mem_stop.wait(0.05):
                            try:
                                if proc.memory_info().rss > memory_limit:
                                    mem_event.set()
                                    if hasattr(signal, "SIGUSR1"):
                                        os.kill(os.getpid(), signal.SIGUSR1)
                                    else:  # pragma: no cover - no signals
                                        _thread.interrupt_main()
                                    break
                            except Exception:
                                break

                    if hasattr(signal, "SIGUSR1"):
                        def _mem_handler(signum, frame):  # pragma: no cover - handler
                            mem_event.set()
                            raise MemoryError("module exceeded memory limit")

                        old_mem_handler = signal.signal(signal.SIGUSR1, _mem_handler)

                    mem_thread = threading.Thread(target=_monitor_mem, daemon=True)
                    mem_thread.start()

                try:
                    if inspect.iscoroutinefunction(fn):
                        coro = fn()
                        if timeout:
                            result = asyncio.run(asyncio.wait_for(coro, timeout))
                        else:
                            result = asyncio.run(coro)
                    else:
                        result = fn()
                        if asyncio.iscoroutine(result):
                            if timeout:
                                result = asyncio.run(asyncio.wait_for(result, timeout))
                            else:
                                result = asyncio.run(result)
                except Exception as exc:  # pragma: no cover - exercise failure path
                    success = False
                    error = str(exc)
                    metrics.crash_count += 1
                    if not safe_mode:
                        raise
                except BaseException:  # pragma: no cover - timeout/memory
                    if mem_event.is_set():
                        success = False
                        error = "module exceeded memory limit"
                        metrics.crash_count += 1
                        if not safe_mode:
                            raise MemoryError(error)
                    elif timeout_event.is_set():
                        success = False
                        error = "module exceeded timeout"
                        metrics.crash_count += 1
                        if not safe_mode:
                            raise TimeoutError(error)
                    else:
                        raise
                finally:
                    if timeout:
                        if old_alarm is not None:
                            signal.setitimer(signal.ITIMER_REAL, 0)
                            signal.signal(signal.SIGALRM, old_alarm)
                        elif timer is not None:
                            timer.cancel()
                    if mem_thread is not None:
                        mem_stop.set()
                        mem_thread.join()
                        if old_mem_handler is not None:
                            signal.signal(signal.SIGUSR1, old_mem_handler)

                    for key, original in old_env.items():
                        if original is None:
                            os.environ.pop(key, None)
                        else:
                            os.environ[key] = original

                    duration = perf_counter() - start

                    if use_psutil_cpu:
                        try:
                            ct = proc.cpu_times()  # type: ignore[union-attr]
                            cpu_after = ct.user + getattr(ct, "system", 0.0)
                        except Exception:
                            cpu_after = cpu_before
                    else:
                        cpu_after = process_time()

                    if use_psutil:
                        try:
                            mem_after = proc.memory_info().rss  # type: ignore[union-attr]
                            mem_peak = mem_after
                        except Exception:
                            mem_after = mem_before
                            mem_peak = mem_after
                    else:
                        mem_after, mem_peak = tracemalloc.get_traced_memory()
                        tracemalloc.stop()

                    module_metric = ModuleMetrics(
                        name=name,
                        duration=duration,
                        cpu_before=cpu_before,
                        cpu_after=cpu_after,
                        cpu_delta=cpu_after - cpu_before,
                        memory_before=mem_before,
                        memory_after=mem_after,
                        memory_delta=mem_after - mem_before,
                        memory_peak=mem_peak,
                        success=success,
                        exception=error,
                        result=result,
                        fixtures=fixtures,
                    )
                    metrics.modules.append(module_metric)

                    # Recursively execute nested workflows returned by the module
                    nested_funcs: list[Callable[[], Any]] = []
                    if isinstance(result, Mapping) and result.get("steps"):
                        for step in result["steps"]:
                            if callable(step):
                                nested_funcs.append(step)
                            elif isinstance(step, Mapping):
                                wf_obj = step.get("workflow") or step.get("call")
                                if callable(wf_obj):
                                    nested_funcs.append(wf_obj)
                    elif isinstance(result, (list, tuple)):
                        nested_funcs = [c for c in result if callable(c)]
                    if nested_funcs:
                        prev_metrics = self.metrics
                        prev_telemetry = self.telemetry
                        nested_metrics = self.run(
                            nested_funcs,
                            safe_mode=safe_mode,
                            test_data=test_data,
                            network_mocks=network_mocks,
                            fs_mocks=fs_mocks,
                            module_fixtures=module_fixtures,
                            roi_delta=roi_delta,
                            timeout=timeout,
                            memory_limit=memory_limit,
                        )
                        metrics.modules.extend(nested_metrics.modules)
                        metrics.crash_count += nested_metrics.crash_count
                        self.metrics = prev_metrics
                        self.telemetry = prev_telemetry

            # ------------------------------------------------------------------
            # Aggregate metrics into a simple telemetry dictionary
            times = {m.name: m.duration for m in metrics.modules}
            cpu_times = {m.name: m.cpu_delta for m in metrics.modules}
            memory_deltas = {m.name: m.memory_delta for m in metrics.modules}
            results = {m.name: m.result for m in metrics.modules}
            fixtures_info: dict[str, Any] = {}
            for m in metrics.modules:
                if m.fixtures:
                    info: dict[str, Any] = {}
                    files = m.fixtures.get("files", {})
                    env = m.fixtures.get("env", {})
                    if files:
                        info["files"] = list(files.keys())
                    if env:
                        info["env"] = dict(env)
                    fixtures_info[m.name] = info
            peak_mem = max((m.memory_peak for m in metrics.modules), default=0)
            peak_per_module = {m.name: m.memory_peak for m in metrics.modules}
            crash_freq = (
                metrics.crash_count / len(metrics.modules)
                if metrics.modules
                else 0.0
            )
            telemetry: dict[str, Any] = {
                "time_per_module": times,
                "cpu_per_module": cpu_times,
                "memory_per_module": memory_deltas,
                "results": results,
                "crash_frequency": crash_freq,
                "peak_memory": peak_mem,
                "peak_memory_per_module": peak_per_module,
            }
            if fixtures_info:
                telemetry["module_fixtures"] = fixtures_info
            if roi_delta is not None:
                telemetry["roi_delta"] = roi_delta

            # Persist metrics inside the sandbox for debugging or analysis.
            try:
                (root / "telemetry.json").write_text(
                    json.dumps(telemetry, default=repr)
                )
            except Exception as exc:
                logger.exception("failed to persist telemetry: %s", exc)

            # Optionally forward metrics to metrics_exporter gauges.
            if _Gauge:
                try:
                    g_time = _Gauge(
                        "workflow_sandbox_module_seconds",
                        "Execution time per module",
                        ["module"],
                    )
                    for mod, dur in times.items():
                        g_time.labels(module=mod).set(dur)

                    g_cpu = _Gauge(
                        "workflow_sandbox_module_cpu_seconds",
                        "CPU time per module",
                        ["module"],
                    )
                    for mod, dur in cpu_times.items():
                        g_cpu.labels(module=mod).set(dur)

                    g_mem_mod = _Gauge(
                        "workflow_sandbox_module_memory_bytes",
                        "Peak memory usage per module",
                        ["module"],
                    )
                    for mod, mem in peak_per_module.items():
                        g_mem_mod.labels(module=mod).set(mem)

                    g_crash = _Gauge(
                        "workflow_sandbox_crash_total",
                        "Crash count",
                    )
                    g_crash.set(metrics.crash_count)

                    g_mem = _Gauge(
                        "workflow_sandbox_peak_memory_bytes",
                        "Peak memory usage",
                    )
                    g_mem.set(peak_mem)

                    if roi_delta is not None:
                        g_roi = _Gauge(
                            "workflow_sandbox_roi_delta",
                            "ROI delta",
                        )
                        g_roi.set(roi_delta)
                except Exception as exc:
                    logger.warning("failed to export metrics: %s", exc, exc_info=True)

            # Forward telemetry to _SandboxMetaLogger if available.
            if _SandboxMetaLogger:
                try:
                    _SandboxMetaLogger(root / "sandbox_meta.log").audit.record(
                        telemetry
                    )
                except Exception as exc:
                    logger.warning("failed to log sandbox metadata: %s", exc, exc_info=True)

            self.metrics = metrics
            self.telemetry = telemetry
            return metrics


def _subprocess_worker(
    conn: multiprocessing.connection.Connection,
    workflow: Any,
    params: dict[str, Any],
) -> None:  # pragma: no cover - subprocess
    """Entry point for subprocess isolated execution."""
    cpu_limit = params.pop("cpu_limit", None)
    memory_limit = params.get("memory_limit")
    cgroup_path = None
    if resource and (cpu_limit or memory_limit):
        try:
            if cpu_limit:
                sec = int(cpu_limit)
                resource.setrlimit(resource.RLIMIT_CPU, (sec, sec))
            if memory_limit:
                resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
        except Exception:
            pass
    elif cpu_limit or memory_limit:
        try:
            from .environment import _cgroup_v2_supported, _create_cgroup

            if _cgroup_v2_supported():
                cgroup_path = _create_cgroup(cpu_limit, memory_limit)
                if cgroup_path is not None:
                    try:
                        with open(cgroup_path / "cgroup.procs", "w", encoding="utf-8") as fh:
                            fh.write(str(os.getpid()))
                    except Exception:
                        pass
        except Exception:
            cgroup_path = None
    try:
        runner = WorkflowSandboxRunner()
        metrics = runner.run(workflow, **params)
        conn.send((metrics, runner.telemetry))
    except Exception as exc:  # pragma: no cover - safety
        conn.send((RunMetrics(), {"error": str(exc), "trace": traceback.format_exc()}))
    finally:
        if cgroup_path is not None:
            try:
                from .environment import _cleanup_cgroup

                _cleanup_cgroup(cgroup_path)
            except Exception:
                pass
        conn.close()


__all__ = [
    "WorkflowSandboxRunner",
    "RunMetrics",
    "ModuleMetrics",
    "EmptyWorkflowError",
]
