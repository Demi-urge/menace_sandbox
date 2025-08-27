from __future__ import annotations

"""Run a series of workflow modules in a sandboxed directory."""

import builtins
import importlib
import inspect
import logging
import os
import tempfile
from types import ModuleType
from typing import Iterable, Callable, Any, Dict, List

from .environment import generate_input_stubs
from .stub_providers import StubProvider


logger = logging.getLogger(__name__)


class WorkflowSandboxRunner:
    """Execute workflow modules sequentially inside an isolated sandbox.

    Parameters
    ----------
    modules:
        Iterable of callables or module names.  Each item represents a step in
        the workflow.  When a module is provided its ``run`` attribute will be
        invoked if present.
    safe_mode:
        When ``True`` network operations are disabled and exceptions from
        modules are captured instead of bubbling up.
    stub_providers:
        Optional list of stub provider callbacks forwarded to
        :func:`generate_input_stubs` for domain specific payloads.
    """

    def __init__(
        self,
        modules: Iterable[str | ModuleType | Callable[..., Any]],
        *,
        safe_mode: bool = False,
        stub_providers: List[StubProvider] | None = None,
    ) -> None:
        self.modules = list(modules)
        self.safe_mode = safe_mode
        self.stub_providers = stub_providers

    # ------------------------------------------------------------------
    def _resolve_path(self, root: str, path: str) -> str:
        if os.path.isabs(path):
            path = path.lstrip(os.sep)
        return os.path.join(root, path)

    def _patch_filesystem(self, root: str) -> List[Callable[[], None]]:
        """Redirect basic filesystem mutations into ``root``."""
        teardowns: List[Callable[[], None]] = []

        original_open = builtins.open

        def sandbox_open(file: str | bytes | os.PathLike[str], mode: str = "r", *a, **kw):
            real_path = self._resolve_path(root, str(file))
            if any(m in mode for m in ("w", "a", "x", "+")):
                os.makedirs(os.path.dirname(real_path), exist_ok=True)
            return original_open(real_path, mode, *a, **kw)

        builtins.open = sandbox_open  # type: ignore[assignment]
        teardowns.append(lambda: setattr(builtins, "open", original_open))

        for name in ["remove", "unlink"]:
            if hasattr(os, name):
                original = getattr(os, name)

                def _wrapped(path, _orig=original):
                    real_path = self._resolve_path(root, path)
                    return _orig(real_path)

                setattr(os, name, _wrapped)
                teardowns.append(lambda n=name, o=original: setattr(os, n, o))

        for name in ["rename", "replace"]:
            if hasattr(os, name):
                original = getattr(os, name)

                def _wrapped(src, dst, _orig=original):
                    src_path = self._resolve_path(root, src)
                    dst_path = self._resolve_path(root, dst)
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    return _orig(src_path, dst_path)

                setattr(os, name, _wrapped)
                teardowns.append(lambda n=name, o=original: setattr(os, n, o))

        return teardowns

    def _patch_network(self) -> List[Callable[[], None]]:
        """Disable network requests when ``safe_mode`` is enabled."""
        teardowns: List[Callable[[], None]] = []
        try:
            import requests  # type: ignore

            original_req = requests.Session.request

            def _blocked(self, *a, **kw):  # pragma: no cover - trivial
                raise RuntimeError("network access disabled in safe_mode")

            requests.Session.request = _blocked  # type: ignore[assignment]
            teardowns.append(lambda: setattr(requests.Session, "request", original_req))
        except Exception:  # pragma: no cover - optional dependency
            pass

        try:
            import httpx  # type: ignore

            original_req = httpx.Client.request

            def _blocked(self, *a, **kw):  # pragma: no cover - trivial
                raise RuntimeError("network access disabled in safe_mode")

            httpx.Client.request = _blocked  # type: ignore[assignment]
            teardowns.append(lambda: setattr(httpx.Client, "request", original_req))
        except Exception:  # pragma: no cover - optional dependency
            pass

        try:
            import urllib.request as urllib_request  # type: ignore

            original_open = urllib_request.urlopen

            def _blocked(*a, **kw):  # pragma: no cover - trivial
                raise RuntimeError("network access disabled in safe_mode")

            urllib_request.urlopen = _blocked  # type: ignore[assignment]
            teardowns.append(lambda: setattr(urllib_request, "urlopen", original_open))
        except Exception:  # pragma: no cover - optional dependency
            pass

        return teardowns

    # ------------------------------------------------------------------
    def _load_callable(self, spec: str | ModuleType | Callable[..., Any]) -> tuple[str, Callable[[], Any]]:
        if isinstance(spec, str):
            module = importlib.import_module(spec)
            func = getattr(module, "run", None)
            if callable(func):
                return module.__name__, func
            raise AttributeError(f"module '{spec}' lacks a 'run' callable")
        if callable(spec):
            return getattr(spec, "__name__", repr(spec)), spec  # type: ignore[return-value]
        if isinstance(spec, ModuleType):
            func = getattr(spec, "run", None)
            if callable(func):
                return spec.__name__, func
            raise AttributeError(f"module '{spec.__name__}' lacks a 'run' callable")
        raise TypeError(f"Unsupported workflow step: {spec!r}")

    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        """Execute the configured workflow and return success metrics."""
        metrics: Dict[str, Any] = {"modules": [], "success": True}

        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "repos"), exist_ok=True)
            os.makedirs(os.path.join(tmpdir, "data"), exist_ok=True)

            teardowns = self._patch_filesystem(tmpdir)
            if self.safe_mode:
                teardowns.extend(self._patch_network())

            try:
                for step in self.modules:
                    name, func = self._load_callable(step)
                    stub: Dict[str, Any] = {}
                    if inspect.signature(func).parameters:
                        try:
                            stubs = generate_input_stubs(
                                1, target=func, providers=self.stub_providers
                            )
                            if stubs:
                                stub = dict(stubs[0])
                        except Exception:
                            logger.exception("stub generation failed for %s", name)
                    logger.info("running %s with stub %s", name, stub)
                    try:
                        result = func(**stub)
                        metrics["modules"].append(
                            {
                                "module": name,
                                "success": True,
                                "result": result,
                                "stub": stub,
                            }
                        )
                    except Exception as exc:  # pragma: no cover - exercise safe paths
                        metrics["modules"].append(
                            {
                                "module": name,
                                "success": False,
                                "error": str(exc),
                                "stub": stub,
                            }
                        )
                        metrics["success"] = False
                        if not self.safe_mode:
                            raise
                return metrics
            finally:
                for td in reversed(teardowns):
                    try:
                        td()
                    except Exception:  # pragma: no cover - best effort cleanup
                        pass

