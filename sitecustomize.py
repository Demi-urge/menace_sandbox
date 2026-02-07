"""Runtime customisations applied on interpreter startup.

In addition to wiring package aliases, this module also patches ``python-dotenv``
to fail gracefully when it encounters malformed ``.env`` files.  Several entry
points load environment variables automatically and historically a single bad
line (for example ``export FOO=bar`` copied from a shell script) would cause the
entire application to abort.  The wrapper implemented here catches the
``DotenvParseError`` exception raised by the library and converts it into a
logged warning so that execution can proceed using any successfully parsed
values.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
import threading
import traceback
import types
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_REPO_ROOT = Path(__file__).resolve().parent
_PARENT = _REPO_ROOT.parent
_NESTED = _REPO_ROOT / "menace_sandbox"
if _PARENT.is_dir() and str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))
if _NESTED.is_dir() and str(_NESTED) not in sys.path:
    sys.path.append(str(_NESTED))

# Ensure bootstrap policy helpers remain importable even when callers prune
# ``sys.path`` aggressively during startup. This registration mirrors the
# behaviour used for ``context_builder_util`` below so that modules importing
# ``bootstrap_timeout_policy`` (or its package-qualified variant) continue to
# resolve the local implementation without relying on the execution context's
# working directory.
_BOOTSTRAP_TIMEOUT_PATH = _REPO_ROOT / "bootstrap_timeout_policy.py"
if _BOOTSTRAP_TIMEOUT_PATH.is_file():
    try:  # pragma: no cover - defensive import wiring
        _btp_spec = importlib.util.spec_from_file_location(
            "bootstrap_timeout_policy", _BOOTSTRAP_TIMEOUT_PATH
        )
        if _btp_spec is not None and _btp_spec.loader is not None:
            _btp_mod = importlib.util.module_from_spec(_btp_spec)
            sys.modules.setdefault("bootstrap_timeout_policy", _btp_mod)
            sys.modules.setdefault("menace_sandbox.bootstrap_timeout_policy", _btp_mod)
            _btp_spec.loader.exec_module(_btp_mod)
    except Exception:
        logging.getLogger(__name__).debug(
            "Failed to register packaged bootstrap_timeout_policy", exc_info=True
        )

_CTX_BUILDER_PATH = _NESTED / "context_builder_util.py"
try:  # pragma: no cover - defensive import wiring
    _ctx_spec = importlib.util.spec_from_file_location(
        "menace_sandbox.context_builder_util", _CTX_BUILDER_PATH
    )
    if _ctx_spec is not None and _ctx_spec.loader is not None:
        _ctx_mod = importlib.util.module_from_spec(_ctx_spec)
        _ctx_spec.loader.exec_module(_ctx_mod)
        sys.modules.setdefault("menace_sandbox.context_builder_util", _ctx_mod)
        sys.modules.setdefault("context_builder_util", _ctx_mod)
except Exception:  # pragma: no cover - avoid bootstrap breakage
    logging.getLogger(__name__).debug(
        "Failed to register packaged context_builder_util", exc_info=True
    )

base = Path(__file__).resolve().parent

# Provide backwards compatibility stubs only when the real modules are missing.
def _register_stub(package: str, module: types.ModuleType, expected: Path) -> None:
    if expected.exists():
        return

    sys.modules.setdefault(package, module)


sandbox_runner_pkg = types.ModuleType("sandbox_runner")
sandbox_runner_pkg.__path__ = [str(base / "sandbox_runner")]
_register_stub("sandbox_runner", sandbox_runner_pkg, base / "sandbox_runner" / "__init__.py")
_register_stub(
    "menace_sandbox.sandbox_runner", sandbox_runner_pkg, base / "sandbox_runner" / "__init__.py"
)

menace_root_pkg = types.ModuleType("menace_sandbox")
menace_root_pkg.__path__ = [str(base)]
_register_stub("menace_sandbox", menace_root_pkg, base / "__init__.py")


def _register_optional_stub(module_name: str, stub_module: str) -> None:
    """Expose ``stub_module`` under ``module_name`` when the real package is missing."""

    if module_name in sys.modules:
        return

    try:  # pragma: no cover - only executed when dependency available
        importlib.import_module(module_name)
    except Exception:  # pragma: no cover - exercised in constrained environments
        stub = importlib.import_module(stub_module)
        sys.modules.setdefault(module_name, stub)


_register_optional_stub("annoy", "annoy_stub")
_register_optional_stub("pydantic", "pydantic_stub")


def _register_sklearn_metrics_shim() -> None:
    """Expose the local sklearn.metrics shim when scikit-learn is unavailable."""

    if "sklearn.metrics" in sys.modules:
        return

    try:  # pragma: no cover - prefer real sklearn when available
        importlib.import_module("sklearn.metrics")
        return
    except Exception:  # pragma: no cover - fall back to local shim
        pass

    try:
        shim = importlib.import_module("sklearn_local.metrics")
    except Exception:  # pragma: no cover - missing local shim
        return

    sys.modules.setdefault("sklearn.metrics", shim)
    sklearn_pkg = sys.modules.get("sklearn")
    if sklearn_pkg is None:
        sklearn_pkg = types.ModuleType("sklearn")
        sklearn_pkg.__path__ = []
        sys.modules["sklearn"] = sklearn_pkg
    setattr(sklearn_pkg, "metrics", shim)


_register_sklearn_metrics_shim()


def _patch_dotenv() -> None:
    """Install tolerant wrappers around ``python-dotenv`` helpers.

    ``python-dotenv`` raises :class:`~dotenv.exceptions.DotenvParseError` when it
    encounters unexpected syntax.  This is correct behaviour for strict setups
    but undesirable for the sandbox where environment files are often edited by
    hand.  We patch both :func:`dotenv.load_dotenv` and
    :func:`dotenv.dotenv_values` to log and continue instead of bubbling the
    exception.
    """

    try:
        import dotenv
        from dotenv.exceptions import DotenvParseError
    except Exception:  # pragma: no cover - optional dependency
        return

    logger = logging.getLogger("menace_sandbox.dotenv")

    original_load = getattr(dotenv, "load_dotenv", None)
    if callable(original_load):

        def _safe_load_dotenv(*args, **kwargs):
            try:
                return original_load(*args, **kwargs)
            except DotenvParseError as exc:  # pragma: no cover - requires bad file
                logger.warning("Failed to parse dotenv file: %s", exc)
                return False

        dotenv.load_dotenv = _safe_load_dotenv  # type: ignore[assignment]

    original_values = getattr(dotenv, "dotenv_values", None)
    if callable(original_values):

        def _safe_dotenv_values(*args, **kwargs):
            try:
                return original_values(*args, **kwargs)
            except DotenvParseError as exc:  # pragma: no cover - requires bad file
                logger.warning("Failed to parse dotenv file: %s", exc)
                return {}

        dotenv.dotenv_values = _safe_dotenv_values  # type: ignore[assignment]


_patch_dotenv()


def _patch_threading_excepthook() -> None:
    """Install a resilient ``threading.excepthook`` implementation."""

    original_hook = getattr(threading, "excepthook", None)
    if not callable(original_hook):  # pragma: no cover - Python < 3.8
        return

    def _safe_thread_name(thread: threading.Thread | None) -> str:
        if thread is None:
            return "thread"
        try:
            name = thread.name
        except Exception:  # pragma: no cover - defensive guard
            name = None
        if name:
            return f"thread {name!r}"
        ident = getattr(thread, "ident", None)
        if ident is not None:
            return f"thread ident={ident}"
        return "thread"

    def _safe_excepthook(args: threading.ExceptHookArgs) -> None:
        try:
            thread_label = _safe_thread_name(getattr(args, "thread", None))
            stream = getattr(sys, "stderr", None)
            if stream is not None:
                try:
                    stream.write(f"Exception in {thread_label}\n")
                except Exception:  # pragma: no cover - extremely defensive
                    pass
            traceback.print_exception(
                getattr(args, "exc_type", None),
                getattr(args, "exc_value", None),
                getattr(args, "exc_traceback", None),
            )
        except Exception:  # pragma: no cover - fallback to default handler
            try:
                original_hook(args)
            except Exception:
                pass

    threading.excepthook = _safe_excepthook


_patch_threading_excepthook()
