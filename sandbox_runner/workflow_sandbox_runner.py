"""Utilities for executing workflows in a sandboxed file system."""
from __future__ import annotations

import builtins
import logging
import tempfile
import pathlib
from typing import Any, Callable, Mapping


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
    ) -> Any:
        """Execute ``workflow_callable`` inside a sandbox.

        Parameters
        ----------
        workflow_callable:
            The callable representing the workflow to execute.
        safe_mode:
            When ``True``, network access is disabled and exceptions raised by
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
        """

        test_data = dict(test_data or {})
        expected_outputs = dict(expected_outputs or {})

        with tempfile.TemporaryDirectory() as tmp:
            original_open = builtins.open
            import pathlib as _pathlib
            original_path = _pathlib.PosixPath

            sandbox_root = _pathlib.Path(tmp)

            def _resolve_path(p: str | _pathlib.Path) -> _pathlib.Path:
                p = original_path(p)
                if p.is_absolute():
                    p = original_path(*p.parts[1:])
                return sandbox_root / p

            # Pre-populate any provided test data into the sandbox.
            for name, content in test_data.items():
                real_path = _resolve_path(name)
                real_path.parent.mkdir(parents=True, exist_ok=True)
                mode = "wb" if isinstance(content, bytes) else "w"
                with original_open(real_path, mode) as fh:
                    fh.write(content)

            def sandbox_open(file: str | bytes | _pathlib.Path, mode: str = "r", *a, **kw):
                path = original_path(file)
                real_path = _resolve_path(path)
                if any(m in mode for m in ("w", "a", "x", "+")):
                    real_path.parent.mkdir(parents=True, exist_ok=True)
                return original_open(real_path, mode, *a, **kw)

            def sandbox_path(*args: Any, **kwargs: Any) -> Path:
                return _resolve_path(original_path(*args, **kwargs))

            builtins.open = sandbox_open
            _original_Path = _pathlib.Path
            _pathlib.Path = sandbox_path  # type: ignore[assignment]

            # Monkeypatch file saving utilities so they operate within the sandbox.
            import os as _os
            import shutil as _shutil

            def _patch_copy(name: str) -> Callable[[], None]:
                original = getattr(_shutil, name)

                def _wrapped(src, dst, *a, **kw):
                    src_path = _resolve_path(src)
                    dst_path = _resolve_path(dst)
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    return original(src_path, dst_path, *a, **kw)

                setattr(_shutil, name, _wrapped)
                return lambda: setattr(_shutil, name, original)

            def _patch_rename(name: str) -> Callable[[], None]:
                original = getattr(_os, name)

                def _wrapped(src, dst, *a, **kw):
                    src_path = _resolve_path(src)
                    dst_path = _resolve_path(dst)
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    return original(src_path, dst_path, *a, **kw)

                setattr(_os, name, _wrapped)
                return lambda: setattr(_os, name, original)

            teardowns: list[Callable[[], None]] = []
            for _name in ["copy", "copy2", "copyfile", "move"]:
                if hasattr(_shutil, _name):
                    teardowns.append(_patch_copy(_name))
            for _name in ["rename", "replace"]:
                if hasattr(_os, _name):
                    teardowns.append(_patch_rename(_name))

            # Network mocking in safe mode.
            if safe_mode:
                def _patch_requests() -> Callable[[], None]:
                    try:
                        import requests  # type: ignore
                    except Exception:  # pragma: no cover - optional dependency
                        return lambda: None

                    original_req = requests.Session.request

                    def _blocked(self, *a, **kw):  # pragma: no cover - trivial
                        raise RuntimeError("network access disabled in safe_mode")

                    requests.Session.request = _blocked  # type: ignore[assignment]
                    return lambda: setattr(requests.Session, "request", original_req)

                def _patch_httpx() -> Callable[[], None]:
                    try:
                        import httpx  # type: ignore
                    except Exception:  # pragma: no cover - optional dependency
                        return lambda: None

                    original_req = httpx.Client.request
                    httpx.Client.request = (  # type: ignore[assignment]
                        lambda self, *a, **kw: (_ for _ in ()).throw(
                            RuntimeError("network access disabled in safe_mode")
                        )
                    )
                    return lambda: setattr(httpx.Client, "request", original_req)

                def _patch_urllib() -> Callable[[], None]:
                    try:
                        import urllib.request as _urllib_request
                    except Exception:  # pragma: no cover - optional dependency
                        return lambda: None

                    original_open = _urllib_request.urlopen

                    def _blocked(*a, **kw):  # pragma: no cover - trivial
                        raise RuntimeError("network access disabled in safe_mode")

                    _urllib_request.urlopen = _blocked  # type: ignore[assignment]
                    return lambda: setattr(_urllib_request, "urlopen", original_open)

                teardowns.extend(
                    [_patch_requests(), _patch_httpx(), _patch_urllib()]
                )

            # Allow callers to inject additional mocks.
            for injector in mock_injectors or []:
                teardowns.append(injector(sandbox_root))

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
                for td in reversed(teardowns):
                    try:
                        td()
                    except Exception:  # pragma: no cover - best effort cleanup
                        pass
                builtins.open = original_open
                _pathlib.Path = _original_Path
