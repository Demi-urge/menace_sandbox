"""High level pipeline orchestrating model automation.

When :mod:`marshmallow` is not installed, a very small schema system is
provided to validate records.  The fallback checks that required fields
are present and match the expected type instead of blindly returning the
data.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .model_automation_dependencies import (
    DB_ROUTER,
    MENACE_ID,
    _LazyAggregator,
    _build_default_hierarchy,
    _build_default_validator,
    _capital_manager_cls,
    _create_synthesis_task,
    _implementation_optimiser_cls,
    _load_research_aggregator,
    _make_research_item,
    _planning_components,
)

if TYPE_CHECKING:  # pragma: no cover - typing only import
    from .shared.model_pipeline_core import ModelAutomationPipeline as _ModelAutomationPipeline
else:  # pragma: no cover - runtime alias populated lazily
    _ModelAutomationPipeline = Any  # type: ignore[misc]


_PIPELINE_CLS: "type[_ModelAutomationPipeline] | None" = None
LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - typing only imports
    from .pre_execution_roi_bot import ROIResult
    from .task_handoff_bot import TaskPackage
else:  # pragma: no cover - runtime fallback avoids circular imports
    ROIResult = Any  # type: ignore
    TaskPackage = Any  # type: ignore


@dataclass
class AutomationResult:
    """Final pipeline output."""

    package: Optional["TaskPackage"]
    roi: Optional["ROIResult"]
    warnings: Dict[str, List[Dict[str, Any]]] | None = None
    workflow_evolution: List[Dict[str, Any]] | None = None


def _load_pipeline_cls() -> "type[_ModelAutomationPipeline]":
    """Import and cache the concrete :class:`ModelAutomationPipeline` implementation."""

    global _PIPELINE_CLS
    if _PIPELINE_CLS is None:
        from .entry_pipeline_loader import load_pipeline_class
        try:
            _PIPELINE_CLS = load_pipeline_class()
        except Exception as exc:
            LOGGER.exception(
                "Failed to load ModelAutomationPipeline",
                extra={"module": __name__},
            )
            traceback_details = traceback.format_exc()
            raise ImportError(
                "ModelAutomationPipeline unavailable: "
                f"{exc}\n{traceback_details}"
            ) from exc
        else:
            globals()["ModelAutomationPipeline"] = _PIPELINE_CLS
    return _PIPELINE_CLS


class _ModelAutomationPipelineProxy:
    """Callable proxy that resolves the pipeline class on first use."""

    def __init__(self, error: BaseException) -> None:
        self._error = error
        self._resolved: "type[_ModelAutomationPipeline] | None" = None

    def _resolve(self) -> "type[_ModelAutomationPipeline]":
        if self._resolved is not None:
            return self._resolved
        try:
            self._resolved = _load_pipeline_cls()
        except Exception as exc:
            LOGGER.exception(
                "Deferred load of ModelAutomationPipeline failed",
                extra={"module": __name__, "initial_error": str(self._error)},
            )
            raise self._error from exc
        return self._resolved

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        cls = self._resolve()
        return cls(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        cls = self._resolve()
        return getattr(cls, name)

    def __repr__(self) -> str:
        return f"<ModelAutomationPipelineProxy error={self._error!r}>"


def get_pipeline_class() -> "type[_ModelAutomationPipeline]":
    """Return the concrete :class:`ModelAutomationPipeline` implementation."""

    return _load_pipeline_cls()


def __getattr__(name: str) -> Any:
    """Provide lazy access to heavy imports to avoid circular import failures."""

    if name == "ModelAutomationPipeline":
        return _load_pipeline_cls()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    """Expose lazily provided attributes to :func:`dir`."""

    return sorted(list(globals().keys()) + ["ModelAutomationPipeline"])


__all__ = ["AutomationResult", "ModelAutomationPipeline", "get_pipeline_class"]


try:  # pragma: no cover - import-time availability hint
    ModelAutomationPipeline = _load_pipeline_cls()
except Exception as exc:  # pragma: no cover - defer failure
    ModelAutomationPipeline = _ModelAutomationPipelineProxy(exc)
