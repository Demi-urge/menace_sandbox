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
    ) -> Any:
        """Execute ``workflow_callable`` inside a sandbox.

        Parameters
        ----------
        workflow_callable:
            The callable representing the workflow to execute.
        safe_mode:
            When ``True``, exceptions raised by the workflow are captured and
            returned instead of being raised.
        test_data:
            Optional mapping of file paths to contents. When a path from this
            mapping is opened for reading, the provided content is returned via
            an in-memory buffer instead of accessing the filesystem.
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

            try:
                if safe_mode:
                    try:
                        return workflow_callable()
                    except Exception as exc:  # pragma: no cover - optional path
                        return exc
                return workflow_callable()
            finally:
                builtins.open = original_open
                _pathlib.Path = _original_Path
