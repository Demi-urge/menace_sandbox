"""Isolated workflow execution with telemetry support.

This module provides :class:`WorkflowSandboxRunner` which runs one or more
callables inside a temporary directory.  File system access is redirected to
that directory and common networking libraries are patched so requests can be
stubbed out.  Each executed callable has execution time, memory usage and
errors recorded and aggregated into a :class:`RunMetrics` instance.
"""

from __future__ import annotations

import builtins
import contextlib
import io
import inspect
import pathlib
import shutil
import tempfile
import urllib.parse
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping
from unittest import mock

try:  # pragma: no cover - psutil is optional
    import psutil  # type: ignore
except Exception:  # pragma: no cover - psutil missing
    psutil = None  # type: ignore

import tracemalloc


# ---------------------------------------------------------------------------
@dataclass
class ModuleMetrics:
    """Telemetry captured for a single executed module."""

    name: str
    duration: float
    memory_before: int
    memory_after: int
    memory_delta: int
    success: bool
    exception: str | None = None


@dataclass
class RunMetrics:
    """Aggregated metrics across all executed modules."""

    modules: list[ModuleMetrics] = field(default_factory=list)
    crash_count: int = 0


# ---------------------------------------------------------------------------
class WorkflowSandboxRunner:
    """Execute workflows inside an isolated sandbox."""

    def __init__(self) -> None:
        self.telemetry: RunMetrics | None = None

    # ------------------------------------------------------------------
    def _resolve(self, root: pathlib.Path, path: str | pathlib.Path) -> pathlib.Path:
        p = pathlib.Path(path)
        if p.is_absolute():
            p = pathlib.Path(*p.parts[1:])
        return root / p

    # ------------------------------------------------------------------
    def run(
        self,
        workflow: Callable[[], Any] | Iterable[Callable[[], Any]],
        *,
        safe_mode: bool = False,
        test_data: Mapping[str, str | bytes] | None = None,
    ) -> RunMetrics:
        """Execute ``workflow`` inside a sandbox and return telemetry.

        ``workflow`` may be a single callable or an iterable of callables.
        ``test_data`` provides stubbed file contents or network responses.  Keys
        containing an ``http`` or ``https`` scheme are treated as URLs and
        returned via monkeypatched networking libraries.  All other keys are
        assumed to represent file paths and are written into the sandbox before
        execution.
        """

        test_data = dict(test_data or {})
        file_data: dict[str, str | bytes] = {}
        network_data: dict[str, str | bytes] = {}
        for key, value in test_data.items():
            scheme = urllib.parse.urlparse(key).scheme
            if scheme in {"http", "https"}:
                network_data[key] = value
            else:
                file_data[key] = value

        proc = psutil.Process() if psutil else None

        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
            root = pathlib.Path(tmp)

            funcs = [workflow] if callable(workflow) else list(workflow)

            # Copy each module's source file into the sandbox for completeness.
            for fn in funcs:
                try:
                    src = inspect.getsourcefile(fn) or inspect.getfile(fn)
                    if src:
                        shutil.copy2(src, root / pathlib.Path(src).name)
                except Exception:  # pragma: no cover - best effort
                    pass

            # ------------------------------------------------------------------
            # Monkeypatch filesystem helpers so all paths resolve inside ``root``
            original_open = builtins.open

            def sandbox_open(file, mode="r", *a, **kw):
                path = self._resolve(root, file)
                if any(m in mode for m in ("w", "a", "x", "+")):
                    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
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

            # Pre-populate any provided file data into the sandbox.
            for name, content in file_data.items():
                real = self._resolve(root, name)
                real.parent.mkdir(parents=True, exist_ok=True)
                mode = "wb" if isinstance(content, (bytes, bytearray)) else "w"
                with original_open(real, mode) as fh:
                    fh.write(content)

            # ------------------------------------------------------------------
            # Monkeypatch common networking libraries

            def _response_for(content: str | bytes):
                data = content if isinstance(content, (bytes, bytearray)) else str(content).encode()

                class _Resp:
                    def __init__(self, b: bytes):
                        self.content = b
                        self.status_code = 200

                    @property
                    def text(self) -> str:
                        return self.content.decode()

                return _Resp(data)

            try:  # pragma: no cover - optional dependency
                import requests  # type: ignore

                orig_request = requests.Session.request

                def fake_request(self, method, url, *a, **kw):
                    if url in network_data:
                        return _response_for(network_data[url])
                    if safe_mode:
                        raise RuntimeError("network access disabled in safe_mode")
                    return orig_request(self, method, url, *a, **kw)

                stack.enter_context(
                    mock.patch.object(requests.Session, "request", fake_request)
                )
            except Exception:  # pragma: no cover
                pass

            try:  # pragma: no cover - optional dependency
                import urllib.request as urllib_request  # type: ignore

                orig_urlopen = urllib_request.urlopen

                def fake_urlopen(url, *a, **kw):
                    u = url if isinstance(url, str) else url.get_full_url()
                    if u in network_data:
                        data = network_data[u]
                        if isinstance(data, str):
                            data = data.encode()
                        return io.BytesIO(data)
                    if safe_mode:
                        raise RuntimeError("network access disabled in safe_mode")
                    return orig_urlopen(url, *a, **kw)

                stack.enter_context(
                    mock.patch.object(urllib_request, "urlopen", fake_urlopen)
                )
            except Exception:  # pragma: no cover
                pass

            metrics = RunMetrics()

            for fn in funcs:
                name = getattr(fn, "__name__", repr(fn))

                start = perf_counter()
                mem_before = mem_after = 0
                use_psutil = bool(proc)
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

                try:
                    fn()
                except Exception as exc:  # pragma: no cover - exercise failure path
                    success = False
                    error = str(exc)
                    metrics.crash_count += 1
                    if not safe_mode:
                        raise
                finally:
                    duration = perf_counter() - start

                    if use_psutil:
                        try:
                            mem_after = proc.memory_info().rss  # type: ignore[union-attr]
                        except Exception:
                            mem_after = mem_before
                    else:
                        mem_after, _ = tracemalloc.get_traced_memory()
                        tracemalloc.stop()

                    module_metric = ModuleMetrics(
                        name=name,
                        duration=duration,
                        memory_before=mem_before,
                        memory_after=mem_after,
                        memory_delta=mem_after - mem_before,
                        success=success,
                        exception=error,
                    )
                    metrics.modules.append(module_metric)

            self.telemetry = metrics
            return metrics


__all__ = ["WorkflowSandboxRunner", "RunMetrics", "ModuleMetrics"]

