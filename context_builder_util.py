"""Helpers for wiring the heavy-weight context builder utilities.

Historically this module relied on :meth:`SourceFileLoader.load_module` to load
``config/create_context_builder.py`` lazily.  ``load_module`` is deprecated and
is notoriously brittle on Windows where concurrent imports occasionally observe
partially initialised modules.  In the sandbox this manifested as intermittent
``ImportError``/``ModuleNotFoundError`` exceptions during bot
internalisation which, in turn, kept the Windows command prompt execution stuck
retrying self-coding bootstrap.

The implementation below uses the modern :mod:`importlib` API together with a
process-wide lock so the module is instantiated exactly once in a thread-safe
fashion.  The resolved module is cached which eliminates repeated filesystem
work and prevents the race conditions that previously left the sandbox waiting
for retries indefinitely.
"""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
import logging
import os
import sys
import threading

_LOGGER = logging.getLogger(__name__)
_MODULE_NAME = "menace_sandbox.config.create_context_builder"
_MODULE_CACHE: ModuleType | None = None
_MODULE_LOCK = threading.Lock()

_DEFAULT_DB_SPEC: tuple[tuple[str, str, str], ...] = (
    ("bots_db", "bots.db", "BOT_DB_PATH"),
    ("code_db", "code.db", "CODE_DB_PATH"),
    ("errors_db", "errors.db", "ERROR_DB_PATH"),
    ("workflows_db", "workflows.db", "WORKFLOW_DB_PATH"),
)


class _ModuleProxy:
    """Lazy proxy exposing the builder module for monkeypatching in tests."""

    def __getattr__(self, name: str) -> object:  # pragma: no cover - trivial proxy
        return getattr(_load_builder_module(), name)

    def __setattr__(self, name: str, value: object) -> None:  # pragma: no cover - trivial proxy
        setattr(_load_builder_module(), name, value)


_create_module: ModuleType | _ModuleProxy = _ModuleProxy()


def _load_builder_module() -> ModuleType:
    """Load and cache the heavy ``create_context_builder`` helper module."""

    global _MODULE_CACHE
    if _MODULE_CACHE is not None:
        return _MODULE_CACHE

    with _MODULE_LOCK:
        if _MODULE_CACHE is not None:
            return _MODULE_CACHE

        module_path = Path(__file__).resolve().parent / "config" / "create_context_builder.py"
        spec = spec_from_file_location(_MODULE_NAME, module_path)
        if spec is None or spec.loader is None:  # pragma: no cover - defensive
            raise ImportError(f"unable to load context builder helpers from {module_path!s}")

        module = module_from_spec(spec)
        # Register the module under both the fully qualified and short names to
        # mirror the behaviour of ``import_compat`` and avoid duplicate imports.
        sys.modules.setdefault(spec.name, module)
        sys.modules.setdefault("config.create_context_builder", module)
        try:
            spec.loader.exec_module(module)
        except Exception:  # pragma: no cover - propagate with context
            _LOGGER.exception("failed to import create_context_builder helper from %s", module_path)
            raise

        _MODULE_CACHE = module
        # Replace the proxy with the resolved module for backwards compatibility
        globals()["_create_module"] = module
        return module


def _expected_db_paths(module: ModuleType) -> dict[str, Path]:
    """Return the database paths expected by ``ContextBuilder``."""

    data_dir = Path(os.getenv("SANDBOX_DATA_DIR", "."))
    spec = getattr(module, "DB_SPEC", _DEFAULT_DB_SPEC)
    paths: dict[str, Path] = {}
    for attr, filename, env_var in spec:
        override = os.getenv(env_var)
        path = Path(override) if override else data_dir / filename
        paths[attr] = path
    return paths


def create_context_builder(*args, **kwargs):  # type: ignore[override]
    """Proxy to :func:`config.create_context_builder.create_context_builder`."""

    module = _load_builder_module()
    ensure_readable = getattr(module, "_ensure_readable", None)
    builder_kwargs: dict[str, str] = {}
    paths = _expected_db_paths(module)
    missing: list[str] = []
    unreadable: list[str] = []
    for attr, path in paths.items():
        candidate: Path | None = path
        if not path.exists():
            missing.append(path.name)
            continue
        if callable(ensure_readable):
            try:
                ensured = ensure_readable(path, path.name)
            except FileNotFoundError:
                missing.append(path.name)
                continue
            except OSError:
                unreadable.append(path.name)
                continue
            else:
                candidate = Path(ensured)

        if not candidate.exists():
            missing.append(path.name)
            continue
        if not candidate.is_file():
            unreadable.append(path.name)
            continue

        builder_kwargs[attr] = str(candidate)

    if missing:
        raise FileNotFoundError(
            f"Missing required context builder database(s): {', '.join(sorted(set(missing)))}"
        )
    if unreadable:
        raise OSError(
            f"Context builder database paths are not files: {', '.join(sorted(set(unreadable)))}"
        )

    stack_overrides: dict[str, object] = {}
    enabled_env = os.getenv("STACK_CONTEXT_ENABLED")
    if enabled_env is not None:
        stack_overrides["enabled"] = enabled_env.strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    top_k_env = os.getenv("STACK_CONTEXT_TOP_K")
    if top_k_env:
        try:
            stack_overrides["top_k"] = int(top_k_env)
        except ValueError:  # pragma: no cover - invalid override
            pass
    langs_env = os.getenv("STACK_CONTEXT_LANGUAGES")
    if langs_env:
        languages = [part.strip() for part in langs_env.split(",") if part.strip()]
        if languages:
            stack_overrides["languages"] = tuple(languages)
    max_lines_env = os.getenv("STACK_CONTEXT_MAX_LINES")
    if max_lines_env:
        try:
            stack_overrides["max_lines"] = int(max_lines_env)
        except ValueError:  # pragma: no cover - invalid override
            pass
    if stack_overrides:
        builder_kwargs["stack_config"] = stack_overrides

    context_builder = getattr(module, "ContextBuilder", None)
    if context_builder is None:
        raise RuntimeError("ContextBuilder helper unavailable")

    try:
        builder = context_builder(**builder_kwargs)
    except TypeError as exc:
        raise ValueError("ContextBuilder implementation rejected database paths") from exc

    return builder


def ensure_fresh_weights(builder) -> None:
    """Refresh context builder weights with basic error handling."""
    try:
        builder.refresh_db_weights()
    except Exception as exc:  # pragma: no cover - simple wrapper
        logging.getLogger(__name__).warning("refresh_db_weights failed: %s", exc)
        raise
