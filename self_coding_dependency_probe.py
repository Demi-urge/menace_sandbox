"""Utility helpers for assessing self-coding dependency readiness.

These helpers are intentionally lightweight so they can be imported in
environments where heavy optional dependencies (``pydantic``, ``sklearn`` and
friends) are absent.  The production sandbox runs on a wide range of Windows
installations where those packages are frequently missing which historically
caused the self-coding bootstrap to loop indefinitely while retrying
internalisation.  Centralising the dependency probes makes it easy for callers
to short-circuit that behaviour before any expensive imports occur.
"""

from __future__ import annotations

from importlib.util import find_spec
from typing import Iterable, Sequence, Tuple
import logging

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

    missing = probe_missing_dependencies(modules)
    return (not missing, missing)

