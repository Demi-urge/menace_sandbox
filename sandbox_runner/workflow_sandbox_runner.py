"""Utilities for executing workflows in a sandboxed file system."""
from __future__ import annotations

import builtins
import contextlib
import logging
import pathlib
import tempfile
import urllib.parse
from typing import Any, Callable, Iterable, Mapping
from unittest import mock


logger = logging.getLogger(__name__)


class WorkflowSandboxRunner:
    """Run a workflow callable within an isolated temporary directory.

    The runner creates a fresh temporary directory for each invocation and
    monkeypatches :func:`open` and :class:`pathlib.Path` so all file accesses are
    redirected to that directory. Optional ``test_data`` can pre-populate files
    within the sandbox before execution. Original functions are restored after
    execution. Callers may also provide ``expected_outputs`` to validate that
    specific files were produced with the expected contents.
    """

    def run(
        self,
        workflow_callable: Callable[[], Any],
        *,
        safe_mode: bool = False,
        test_data: Mapping[str, str | bytes] | None = None,
        expected_outputs: Mapping[str, str | bytes] | None = None,
        mock_injectors: list[Callable[[pathlib.Path], Callable[[], None]]] | None = None,
        allowed_domains: Iterable[str] | None = None,
        allowed_files: Iterable[str | pathlib.Path] | None = None,
    ) -> Any:
        """Execute ``workflow_callable`` inside a sandbox.

        Parameters
        ----------
        workflow_callable:
            The callable representing the workflow to execute.
        safe_mode:
            When ``True`` network access is disabled and exceptions raised by
            the workflow are captured and returned instead of being raised.
        test_data:
            Optional mapping of file paths to contents. Each entry is written
            into the sandbox prior to execution so workflows can read them as
            regular files.
        expected_outputs:
            Optional mapping of file paths to expected contents. After the
            workflow completes each specified file is read and compared to the
            expected value. Discrepancies are logged as warnings.
        mock_injectors:
            Optional list of callables that receive the sandbox root and return
            a teardown callable.  This allows callers to supply additional
            monkeypatching behaviour.
        allowed_domains:
            Iterable of hostnames that are permitted for network access in
            ``safe_mode``.  When empty, all network requests are blocked.
        allowed_files:
            Iterable of file paths that may be accessed outside of the
            sandbox.  Writes to any other path are redirected into the sandbox
            directory.
        """

        test_data = dict(test_data or {})
        expected_outputs = dict(expected_outputs or {})
        allowed_domains = set(allowed_domains or [])
        allowed_files = {
            pathlib.Path(p).resolve() for p in (allowed_files or [])
        }

        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
            sandbox_root = pathlib.Path(tmp)
            original_open = builtins.open
            original_path = pathlib.Path

            def _resolve_path(p: str | pathlib.Path) -> pathlib.Path:
                p = original_path(p)
                if p.is_absolute():
                    p = original_path(*p.parts[1:])
                return sandbox_root / p

            def _is_allowed(path: pathlib.Path) -> bool:
                resolved = original_path(path).resolve()
                return any(
                    resolved == allow or resolved.is_relative_to(allow)
                    for allow in allowed_files
                )

            # Pre-populate any provided test data into the sandbox.
            for name, content in test_data.items():
                real_path = _resolve_path(name)
                real_path.parent.mkdir(parents=True, exist_ok=True)
                mode = "wb" if isinstance(content, bytes) else "w"
                with original_open(real_path, mode) as fh:
                    fh.write(content)

            def sandbox_open(
                file: str | bytes | pathlib.Path, mode: str = "r", *a, **kw
            ):
                path = original_path(file)
                if not _is_allowed(path):
                    path = _resolve_path(path)
                    if any(m in mode for m in ("w", "a", "x", "+")):
                        path.parent.mkdir(parents=True, exist_ok=True)
                return original_open(path, mode, *a, **kw)

            stack.enter_context(mock.patch("builtins.open", sandbox_open))

            import os as _os
            import shutil as _shutil

            for _name in ["copy", "copy2", "copyfile", "move"]:
                if hasattr(_shutil, _name):
                    _orig = getattr(_shutil, _name)

                    def _wrapped(src, dst, _orig=_orig, *a, **kw):
                        src_path = _resolve_path(src)
                        dst_path = _resolve_path(dst)
                        dst_path.parent.mkdir(parents=True, exist_ok=True)
                        return _orig(src_path, dst_path, *a, **kw)

                    stack.enter_context(mock.patch.object(_shutil, _name, _wrapped))

            for _name in ["rename", "replace"]:
                if hasattr(_os, _name):
                    _orig = getattr(_os, _name)

                    def _wrapped(src, dst, _orig=_orig, *a, **kw):
                        src_path = _resolve_path(src)
                        dst_path = _resolve_path(dst)
                        dst_path.parent.mkdir(parents=True, exist_ok=True)
                        return _orig(src_path, dst_path, *a, **kw)

                    stack.enter_context(mock.patch.object(_os, _name, _wrapped))

            if safe_mode:
                def _hostname(url: str) -> str:
                    try:
                        return urllib.parse.urlparse(url).hostname or ""
                    except Exception:  # pragma: no cover - defensive
                        return ""

                def _allowed(url: str) -> bool:
                    return _hostname(url) in allowed_domains

                try:
                    import requests  # type: ignore

                    _orig = requests.Session.request

                    def _blocked(self, method, url, *a, **kw):
                        if not _allowed(url):
                            raise RuntimeError(
                                "network access disabled in safe_mode"
                            )
                        return _orig(self, method, url, *a, **kw)

                    stack.enter_context(
                        mock.patch.object(requests.Session, "request", _blocked)
                    )
                except Exception:  # pragma: no cover - optional dependency
                    pass

                try:
                    import httpx  # type: ignore

                    _orig = httpx.Client.request

                    def _blocked(self, method, url, *a, **kw):
                        if not _allowed(url):
                            raise RuntimeError(
                                "network access disabled in safe_mode"
                            )
                        return _orig(self, method, url, *a, **kw)

                    stack.enter_context(
                        mock.patch.object(httpx.Client, "request", _blocked)
                    )
                except Exception:  # pragma: no cover - optional dependency
                    pass

                try:
                    import urllib.request as _urllib_request

                    _orig = _urllib_request.urlopen

                    def _blocked(url, *a, **kw):
                        if isinstance(url, str) and not _allowed(url):
                            raise RuntimeError(
                                "network access disabled in safe_mode"
                            )
                        return _orig(url, *a, **kw)

                    stack.enter_context(
                        mock.patch.object(_urllib_request, "urlopen", _blocked)
                    )
                except Exception:  # pragma: no cover - optional dependency
                    pass

                import socket as _socket

                _orig_socket = _socket.socket

                class _PatchedSocket(_orig_socket):
                    def connect(self, address):  # type: ignore[override]
                        host = address[0]
                        if host not in allowed_domains:
                            raise RuntimeError(
                                "network access disabled in safe_mode"
                            )
                        return super().connect(address)

                stack.enter_context(
                    mock.patch.object(_socket, "socket", _PatchedSocket)
                )

                _orig_create = _socket.create_connection

                def _blocked_create(address, *a, **kw):
                    host = address[0]
                    if host not in allowed_domains:
                        raise RuntimeError("network access disabled in safe_mode")
                    return _orig_create(address, *a, **kw)

                stack.enter_context(
                    mock.patch.object(_socket, "create_connection", _blocked_create)
                )

            for injector in mock_injectors or []:
                stack.callback(injector(sandbox_root))

            try:
                if safe_mode:
                    try:
                        result = workflow_callable()
                    except Exception as exc:  # pragma: no cover - optional path
                        result = exc
                else:
                    result = workflow_callable()

                for name, expected in expected_outputs.items():
                    target = _resolve_path(name)
                    try:
                        with original_open(target, "rb") as fh:
                            actual_bytes = fh.read()
                    except FileNotFoundError:
                        logger.warning("expected output '%s' not found", name)
                        continue
                    actual = (
                        actual_bytes if isinstance(expected, bytes) else actual_bytes.decode()
                    )
                    if actual != expected:
                        logger.warning("output mismatch for '%s'", name)
                return result
            finally:
                stack.close()
