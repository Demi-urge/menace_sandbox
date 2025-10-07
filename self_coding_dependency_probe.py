"""Utility helpers for assessing self-coding dependency readiness.

These helpers default to lightweight ``find_spec`` probes so they can be
imported in environments where heavy optional dependencies (``pydantic``,
``sklearn`` and friends) are absent.  The production sandbox runs on a wide
range of Windows installations where those packages are frequently missing or
partially installed which historically caused the self-coding bootstrap to loop
indefinitely while retrying internalisation.  Centralising the dependency
probes – and performing targeted runtime imports for brittle components – makes
it easy for callers to short-circuit that behaviour before the full
initialisation sequence starts.
"""

from __future__ import annotations

from importlib.util import find_spec
from functools import lru_cache
from typing import Iterable, Sequence, Tuple
import importlib
import logging
import re

logger = logging.getLogger(__name__)

# Dependencies that are required for the autonomous self-coding stack to
# operate.  ``pydantic`` and ``pydantic_settings`` are needed for
# ``SandboxSettings`` while the ``sklearn`` modules back the quick fix engine.
_DEFAULT_MODULES: Tuple[str, ...] = (
    "pydantic",
    "pydantic_settings",
    "sklearn",
    "sklearn.cluster",
    "sklearn.feature_extraction.text",
)

# Modules that should be imported eagerly to ensure compiled extensions are
# usable.  ``quick_fix_engine`` is notoriously sensitive to partial Windows
# installations where ``find_spec`` succeeds but importing the module fails due
# to missing DLLs.  Probing the import early allows us to surface the precise
# dependency gap instead of looping retries during bot internalisation.
_RUNTIME_IMPORTS: Tuple[str, ...] = ("menace_sandbox.quick_fix_engine",)


_MISSING_MODULE_RE = re.compile(r"No module named ['\"]([^'\"]+)['\"]")
_DLL_LOAD_FAILED_RE = re.compile(r"while importing (?P<module>[^:]+)")
_FAILED_IMPORT_RE = re.compile(r"failed to import (?P<module>[\w.]+)", re.IGNORECASE)


def _iter_exception_chain(exc: BaseException):
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _normalise_module_name(name: str | None) -> tuple[str, ...]:
    if not name:
        return tuple()
    cleaned = name.strip().strip("'\"")
    if not cleaned:
        return tuple()
    parts = cleaned.split(".")
    aliases = {cleaned}
    if parts:
        aliases.add(parts[-1])
    return tuple(sorted(alias for alias in aliases if alias))


def _collect_missing_from_exception(exc: BaseException) -> set[str]:
    missing: set[str] = set()
    for item in _iter_exception_chain(exc):
        message = str(item)
        if isinstance(item, ModuleNotFoundError):
            name = getattr(item, "name", None)
            if not name and item.args:
                match = _MISSING_MODULE_RE.search(message)
                if match:
                    name = match.group(1)
            missing.update(_normalise_module_name(name))
            continue
        if isinstance(item, ImportError):
            missing.update(_normalise_module_name(getattr(item, "name", None)))
        for regex in (_MISSING_MODULE_RE, _DLL_LOAD_FAILED_RE, _FAILED_IMPORT_RE):
            match = regex.search(message)
            if match:
                key = match.groupdict().get("module") or match.group(1)
                missing.update(_normalise_module_name(key))
    return missing


@lru_cache(maxsize=1)
def _runtime_dependency_issues(
    targets: Tuple[str, ...] = _RUNTIME_IMPORTS,
) -> Tuple[str, ...]:
    missing: set[str] = set()
    for module in targets:
        try:
            importlib.import_module(module)
        except Exception as exc:  # pragma: no cover - environment specific
            extracted = _collect_missing_from_exception(exc)
            extracted.update(_normalise_module_name(module))
            missing.update(extracted)
    return tuple(sorted(missing))


def _module_missing(name: str) -> bool:
    """Return ``True`` when *name* cannot be resolved via import machinery."""

    try:
        return find_spec(name) is None
    except Exception as exc:  # pragma: no cover - defensive best effort
        logger.debug("dependency probe failed for %s: %s", name, exc, exc_info=True)
        return True


def probe_missing_dependencies(
    modules: Iterable[str] | None = None,
) -> Sequence[str]:
    """Return a stable sequence of modules that are currently unavailable."""

    target = tuple(modules) if modules is not None else _DEFAULT_MODULES
    missing = {name for name in target if _module_missing(name)}
    return tuple(sorted(missing))


def ensure_self_coding_ready(
    modules: Iterable[str] | None = None,
) -> tuple[bool, Sequence[str]]:
    """Check whether the runtime has the dependencies required for self-coding.

    ``modules`` can override the default probe list which is useful for unit
    tests or specialised sandboxes.  The function returns a ``(ready, missing)``
    tuple so callers can branch on the boolean and log the precise set of
    missing packages when needed.
    """

    missing = set(probe_missing_dependencies(modules))
    if modules is None:
        missing.update(_runtime_dependency_issues())
    return (not missing, tuple(sorted(missing)))

