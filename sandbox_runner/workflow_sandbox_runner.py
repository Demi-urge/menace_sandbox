"""Utilities for executing workflows in a sandboxed file system."""
from __future__ import annotations

import builtins
import io
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping


class WorkflowSandboxRunner:
    """Run a workflow callable within an isolated temporary directory.

    The runner creates a fresh temporary directory for each invocation and
    monkeypatches :func:`open` and :class:`pathlib.Path` so all file accesses are
    redirected to that directory. Optional ``test_data`` can provide in-memory
    file contents for reads. Original functions are restored after execution.
    """

    def run(
        self,
        workflow_callable: Callable[[], Any],
        *,
        safe_mode: bool = False,
        test_data: Mapping[str, str | bytes] | None = None,
        mock_injectors: list[Callable[[Path], Callable[[], None]]] | None = None,
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
            Optional mapping of file paths to contents. When a path from this
            mapping is opened for reading, the provided content is returned via
            an in-memory buffer instead of accessing the filesystem.
        mock_injectors:
            Optional list of callables that receive the sandbox root and return
            a teardown callable.  This allows callers to supply additional
            monkeypatching behaviour.
        """

        test_data = dict(test_data or {})

        with tempfile.TemporaryDirectory() as tmp:
            original_open = builtins.open
            original_path = Path

            sandbox_root = Path(tmp)

            def _resolve_path(p: str | Path) -> Path:
                p = original_path(p)
                if p.is_absolute():
                    p = original_path(*p.parts[1:])
                return sandbox_root / p

            def sandbox_open(file: str | bytes | Path, mode: str = "r", *a, **kw):
                path = original_path(file)
                key = str(path)
                data = None
                if "r" in mode and key in test_data:
                    data = test_data[key]
                elif "r" in mode and path.name in test_data:
                    data = test_data[path.name]
                if data is not None:
                    if "b" in mode:
                        return io.BytesIO(data if isinstance(data, bytes) else data.encode())
                    return io.StringIO(data if isinstance(data, str) else data.decode())
                real_path = _resolve_path(path)
                if any(m in mode for m in ("w", "a", "x", "+")):
                    real_path.parent.mkdir(parents=True, exist_ok=True)
                return original_open(real_path, mode, *a, **kw)

            def sandbox_path(*args: Any, **kwargs: Any) -> Path:
                return _resolve_path(original_path(*args, **kwargs))

            builtins.open = sandbox_open
            import pathlib as _pathlib

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
                        return workflow_callable()
                    except Exception as exc:  # pragma: no cover - optional path
                        return exc
                return workflow_callable()
            finally:
                for td in reversed(teardowns):
                    try:
                        td()
                    except Exception:  # pragma: no cover - best effort cleanup
                        pass
                builtins.open = original_open
                _pathlib.Path = _original_Path
