"""Runtime helpers for loading the automation pipeline implementation.

This module provides a single function that performs the heavyweight import
for :class:`menace_sandbox.shared.pipeline_base.ModelAutomationPipeline`.  It is
kept separate from :mod:`menace_sandbox.shared.model_pipeline_core` so that
imports within the ``shared`` package do not re-enter :mod:`pipeline_base`
while it is still initialising, preventing circular import failures.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - used only for type checking
    from .shared.pipeline_base import ModelAutomationPipeline as _ModelAutomationPipeline
else:  # pragma: no cover - runtime fallback avoids circular import
    _ModelAutomationPipeline = Any  # type: ignore[misc, assignment]

__all__ = ["load_pipeline_class"]


def _resolve_pipeline_module() -> Any:
    """Import ``pipeline_base`` handling partially initialised modules."""

    module_name = "menace_sandbox.shared.pipeline_base"
    module = sys.modules.get(module_name)
    if module is not None and getattr(module, "ModelAutomationPipeline", None) is not None:
        return module

    try:
        module = importlib.import_module(module_name)
    except (ModuleNotFoundError, KeyError):
        module = _import_pipeline_via_spec(module_name)
    pipeline_cls = getattr(module, "ModelAutomationPipeline", None)
    if pipeline_cls is not None:
        return module

    # The module is present but still initialising.  Wait briefly for the
    # attribute to appear before attempting a clean re-import.  This situation
    # occurs when ``pipeline_base`` is imported indirectly while one of its
    # dependencies is still importing ``capital_management_bot``.
    for _ in range(10):
        time.sleep(0.05)
        pipeline_cls = getattr(module, "ModelAutomationPipeline", None)
        if pipeline_cls is not None:
            return module

    # Last resort: remove the partially initialised module and import again.
    sys.modules.pop(module_name, None)
    try:
        module = importlib.import_module(module_name)
    except (ModuleNotFoundError, KeyError):
        module = _import_pipeline_via_spec(module_name)
    return module


def _import_pipeline_via_spec(module_name: str) -> ModuleType:
    """Load ``pipeline_base`` directly from disk when package imports fail."""

    package_root = Path(__file__).resolve().parent
    package_name = "menace_sandbox"

    # Ensure the top-level package is registered so relative imports resolve.
    package_module = sys.modules.get(package_name)
    if package_module is None:
        spec = importlib.util.spec_from_file_location(
            package_name, package_root / "__init__.py"
        )
        if spec is None or spec.loader is None:
            raise ImportError("unable to resolve menace_sandbox package")
        package_module = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = package_module
        spec.loader.exec_module(package_module)

    # Namespace packages such as ``menace_sandbox.shared`` need an explicit
    # module object when the package is loaded via ``spec_from_file_location``.
    shared_name = f"{package_name}.shared"
    shared_module = sys.modules.get(shared_name)
    if shared_module is None:
        shared_module = ModuleType(shared_name)
        shared_module.__path__ = [str(package_root / "shared")]
        shared_module.__package__ = shared_name
        sys.modules[shared_name] = shared_module

    pipeline_path = package_root / "shared" / "pipeline_base.py"
    spec = importlib.util.spec_from_file_location(module_name, pipeline_path)
    if spec is None or spec.loader is None:
        raise ImportError("unable to resolve pipeline_base module")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_pipeline_class() -> "type[_ModelAutomationPipeline]":
    """Return the concrete :class:`ModelAutomationPipeline` implementation."""

    module = _resolve_pipeline_module()
    pipeline_cls = getattr(module, "ModelAutomationPipeline", None)
    if pipeline_cls is None:
        raise ImportError("ModelAutomationPipeline is unavailable")
    return pipeline_cls
