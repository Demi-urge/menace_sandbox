"""Utility helpers for working with the optional :mod:`PyYAML` dependency.

This repository relies on :func:`yaml.safe_load`/``safe_dump`` in a handful of
modules but the pared down execution environment used for automated tests does
not ship with PyYAML installed.  Importing those modules would previously raise
``ModuleNotFoundError`` at import time which in turn caused seemingly unrelated
failures such as ``AttributeError: SelfImprovementEngine`` when the
``self_improvement.engine`` module short-circuited during its import.

``get_yaml`` mirrors the :mod:`yaml` interface we depend on and transparently
falls back to a small shim that raises a descriptive ``ModuleNotFoundError``
when any YAML functionality is exercised.  This allows modules to be imported
successfully so that callers receive an actionable error message at the point of
use rather than an obscure attribute lookup failure during bootstrapping.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any
import logging
import sys

__all__ = ["get_yaml"]


_LOGGER = logging.getLogger(__name__)


def _build_stub(component: str, exc: ModuleNotFoundError) -> ModuleType:
    """Create a lightweight substitute exposing the expected YAML API surface."""

    class _MissingYAMLError(ModuleNotFoundError):
        """Exception raised when YAML functionality is requested without PyYAML."""

        def __init__(self) -> None:
            super().__init__(
                "PyYAML is required for {component}; install menace_sandbox[yaml]"
                " or add PyYAML to the environment.".format(component=component)
            )

    def _not_available(*_a: Any, **_k: Any) -> None:
        raise _MissingYAMLError() from exc

    module = ModuleType("yaml")
    module.safe_load = _not_available  # type: ignore[attr-defined]
    module.safe_dump = _not_available  # type: ignore[attr-defined]
    module.load = _not_available  # type: ignore[attr-defined]
    module.dump = _not_available  # type: ignore[attr-defined]
    module.YAMLError = _MissingYAMLError  # type: ignore[attr-defined]
    module.__all__ = ["safe_load", "safe_dump", "load", "dump", "YAMLError"]
    return module


def get_yaml(component: str, *, warn: bool = True):
    """Return the :mod:`yaml` module or a descriptive fallback stub.

    Parameters
    ----------
    component:
        Human readable name of the component importing YAML.  This is used in
        the fallback error message to make it obvious where PyYAML support is
        required.
    warn:
        When ``True`` a warning is emitted the first time the fallback is used.
    """

    try:  # pragma: no cover - exercise relies on optional dependency
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - executed without PyYAML
        if warn:
            _LOGGER.warning(
                "PyYAML unavailable for %s; YAML features will raise informative errors.",
                component,
                exc_info=exc,
            )
        stub = _build_stub(component, exc)
        sys.modules.setdefault("yaml", stub)
        return stub

    return yaml


if "yaml" not in sys.modules:  # pragma: no cover - import side effect
    try:
        import yaml  # type: ignore  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover - minimal environments
        sys.modules["yaml"] = _build_stub("yaml_fallback", exc)

