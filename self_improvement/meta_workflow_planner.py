from __future__ import annotations

"""Compatibility shim exposing :mod:`meta_workflow_planner` inside the
``self_improvement`` package.

The legacy manual bootstrap process imports
``menace_sandbox.self_improvement.meta_workflow_planner`` even though the
implementation historically lived at the top level of the repository.  When the
package is executed from a flat layout (as happens on Windows manual bootstrap
runs) Python is unable to resolve that fully-qualified module name which in turn
breaks optional meta-planning features.

This module mirrors the public API of the actual implementation by importing it
from one of several candidate module paths and re-exporting its ``__all__``.  By
registering the loaded implementation under the current module name we ensure
subsequent imports receive the fully featured planner without duplicating the
heavy initialisation logic.
"""

from importlib import import_module
from types import ModuleType
from typing import Iterable
import sys

_CANDIDATE_MODULES: tuple[str, ...] = (
    "menace_sandbox.meta_workflow_planner",
    "menace.meta_workflow_planner",
    "meta_workflow_planner",
)


def _load_implementation(candidates: Iterable[str]) -> ModuleType:
    """Return the first successfully imported implementation module."""

    last_error: Exception | None = None
    for name in candidates:
        try:
            return import_module(name)
        except Exception as exc:  # pragma: no cover - best effort logging only
            last_error = exc
    raise ModuleNotFoundError(
        "Unable to locate meta_workflow_planner implementation."
    ) from last_error


_impl = _load_implementation(_CANDIDATE_MODULES)

# Mirror the public API exported by the real implementation.
__all__ = getattr(_impl, "__all__", [])  # type: ignore[assignment]
for symbol in __all__:
    globals()[symbol] = getattr(_impl, symbol)

# Ensure subsequent imports of the shim resolve to the shared implementation.
sys.modules[__name__] = _impl
