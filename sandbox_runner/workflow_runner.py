"""Sandboxed workflow runner.

This module provides :class:`WorkflowSandboxRunner` which executes a single
workflow callable inside an isolated temporary directory.  File system
interactions such as :func:`open` or :class:`pathlib.Path.write_text` are
monkeypatched so that reads and writes are confined to the sandbox.  When the
``safe_mode`` flag is enabled, common networking libraries are also patched to
raise :class:`RuntimeError` on outbound requests.

The runner exposes a simple API::

    runner = WorkflowSandboxRunner()
    result, telemetry = runner.run(workflow)

``telemetry`` contains information about the execution such as duration,
success and captured errors.  The most recent telemetry is also available via
``runner.telemetry``.
"""

from __future__ import annotations

import builtins
import contextlib
import logging
import os
import pathlib
import shutil
import tempfile
from time import perf_counter
from typing import Any, Callable, Mapping
from unittest import mock

try:  # pragma: no cover - optional dependency
    import psutil  # type: ignore
except Exception:  # pragma: no cover - psutil not installed
    psutil = None  # type: ignore


logger = logging.getLogger(__name__)


class WorkflowSandboxRunner:
    """Run a workflow callable within an isolated sandbox.

    Each invocation of :meth:`run` creates a fresh temporary directory and
    redirects basic file and network operations into that directory.  The
    callable is executed with all files confined to the sandbox so the host
    system remains untouched.
    """

    def __init__(self) -> None:
        self.telemetry: dict[str, Any] = {}

    # ------------------------------------------------------------------
    def _resolve(self, root: pathlib.Path, path: str | os.PathLike[str]) -> pathlib.Path:
        p = pathlib.Path(path)
        if p.is_absolute():
            p = pathlib.Path(*p.parts[1:])
        return root / p

    # ------------------------------------------------------------------
    def _patch_filesystem(self, root: pathlib.Path, stack: contextlib.ExitStack) -> None:
        """Redirect basic filesystem calls into ``root``."""

        original_open = builtins.open

        def sandbox_open(file, mode="r", *a, **kw):
            path = self._resolve(root, file)
            if any(m in mode for m in ("w", "a", "x", "+")):
                path.parent.mkdir(parents=True, exist_ok=True)
            return original_open(path, mode, *a, **kw)

        stack.enter_context(mock.patch("builtins.open", sandbox_open))

        original_path_open = pathlib.Path.open
        original_write_text = pathlib.Path.write_text
        original_read_text = pathlib.Path.read_text
        original_write_bytes = pathlib.Path.write_bytes
        original_read_bytes = pathlib.Path.read_bytes

        def path_open(path_obj, *a, **kw):
            mode = a[0] if a else kw.get("mode", "r")
            path = self._resolve(root, path_obj)
            if any(m in mode for m in ("w", "a", "x", "+")):
                path.parent.mkdir(parents=True, exist_ok=True)
            return original_path_open(path, *a, **kw)

        def path_write_text(path_obj, data, *a, **kw):
            path = self._resolve(root, path_obj)
            path.parent.mkdir(parents=True, exist_ok=True)
            return original_write_text(path, data, *a, **kw)

        def path_read_text(path_obj, *a, **kw):
            path = self._resolve(root, path_obj)
            return original_read_text(path, *a, **kw)

        def path_write_bytes(path_obj, data, *a, **kw):
            path = self._resolve(root, path_obj)
            path.parent.mkdir(parents=True, exist_ok=True)
            return original_write_bytes(path, data, *a, **kw)

        def path_read_bytes(path_obj, *a, **kw):
            path = self._resolve(root, path_obj)
            return original_read_bytes(path, *a, **kw)

        stack.enter_context(mock.patch.object(pathlib.Path, "open", path_open))
        stack.enter_context(mock.patch.object(pathlib.Path, "write_text", path_write_text))
        stack.enter_context(mock.patch.object(pathlib.Path, "read_text", path_read_text))
        stack.enter_context(mock.patch.object(pathlib.Path, "write_bytes", path_write_bytes))
        stack.enter_context(mock.patch.object(pathlib.Path, "read_bytes", path_read_bytes))

        # Patch os and shutil helpers that mutate the filesystem
        def wrap_os(name):
            if hasattr(os, name):
                original = getattr(os, name)

                def _wrapped(path, *a, _orig=original, **kw):
                    real = self._resolve(root, path)
                    if name in {"rename", "replace"}:
                        real.parent.mkdir(parents=True, exist_ok=True)
                    return _orig(real, *a, **kw)

                stack.enter_context(mock.patch.object(os, name, _wrapped))

        for n in ["remove", "unlink", "rename", "replace"]:
            wrap_os(n)

        def wrap_shutil(name):
            if hasattr(shutil, name):
                original = getattr(shutil, name)

                def _wrapped(src, dst, *a, _orig=original, **kw):
                    src_path = self._resolve(root, src)
                    dst_path = self._resolve(root, dst)
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    return _orig(src_path, dst_path, *a, **kw)

                stack.enter_context(mock.patch.object(shutil, name, _wrapped))

        for n in ["copy", "copy2", "copyfile", "move"]:
            wrap_shutil(n)

    # ------------------------------------------------------------------
    def _patch_network(self, stack: contextlib.ExitStack) -> None:
        """Disable network requests when ``safe_mode`` is enabled."""

        try:  # pragma: no cover - optional dependency
            import requests  # type: ignore

            def _blocked(self, *a, **kw):
                raise RuntimeError("network access disabled in safe_mode")

            stack.enter_context(
                mock.patch.object(requests.Session, "request", _blocked)
            )
        except Exception:  # pragma: no cover
            pass

        try:  # pragma: no cover - optional dependency
            import httpx  # type: ignore

            def _blocked(self, *a, **kw):
                raise RuntimeError("network access disabled in safe_mode")

            stack.enter_context(mock.patch.object(httpx.Client, "request", _blocked))
        except Exception:  # pragma: no cover
            pass

        try:  # pragma: no cover - optional dependency
            import urllib.request as urllib_request  # type: ignore

            def _blocked(*a, **kw):
                raise RuntimeError("network access disabled in safe_mode")

            stack.enter_context(
                mock.patch.object(urllib_request, "urlopen", _blocked)
            )
        except Exception:  # pragma: no cover
            pass

        try:  # pragma: no cover - optional dependency
            import socket  # type: ignore

            class _PatchedSocket(socket.socket):
                def connect(self, address):  # type: ignore[override]
                    raise RuntimeError("network access disabled in safe_mode")

            stack.enter_context(mock.patch.object(socket, "socket", _PatchedSocket))

            def _blocked_create(*a, **kw):
                raise RuntimeError("network access disabled in safe_mode")

            stack.enter_context(
                mock.patch.object(socket, "create_connection", _blocked_create)
            )
        except Exception:  # pragma: no cover
            pass

    # ------------------------------------------------------------------
    def run(
        self,
        workflow: Callable[[], Any],
        *,
        safe_mode: bool = False,
        test_data: Mapping[str, str | bytes] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Execute ``workflow`` inside a sandbox and return result and telemetry."""

        test_data = dict(test_data or {})

        start = perf_counter()
        proc = psutil.Process() if psutil else None
        mem_start = proc.memory_info().rss if proc else 0

        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
            root = pathlib.Path(tmp)

            # Patch filesystem and optional network behaviour
            self._patch_filesystem(root, stack)
            if safe_mode:
                self._patch_network(stack)

            # Pre-populate any test data
            original_open = builtins.open
            for name, content in test_data.items():
                real = self._resolve(root, name)
                real.parent.mkdir(parents=True, exist_ok=True)
                mode = "wb" if isinstance(content, (bytes, bytearray)) else "w"
                with original_open(real, mode) as fh:
                    fh.write(content)

            success = True
            error: str | None = None
            result: Any | None = None

            try:
                result = workflow()
            except Exception as exc:  # pragma: no cover - exercise failure paths
                success = False
                error = str(exc)
                if not safe_mode:
                    raise

        end = perf_counter()
        mem_end = proc.memory_info().rss if proc else mem_start
        mem_peak_raw = mem_end
        if proc:
            try:  # pragma: no cover - platform specific
                info = proc.memory_info()
                mem_end = info.rss
                mem_peak_raw = getattr(info, "peak_wset", getattr(info, "peak_rss", mem_end))
            except Exception:
                pass

        telemetry = {
            "success": success,
            "error": error,
            "duration": end - start,
            "memory_delta": max(mem_end - mem_start, 0) / (1024 * 1024),
            "memory_peak": max(mem_peak_raw - mem_start, 0) / (1024 * 1024),
        }
        self.telemetry = telemetry
        return result, telemetry


__all__ = ["WorkflowSandboxRunner"]

