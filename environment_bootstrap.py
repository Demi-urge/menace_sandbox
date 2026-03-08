"""Legacy compatibility wrapper for bootstrap imports.

Prefer importing from ``menace.environment_bootstrap``. This root-level module
is retained for external callers that still import ``environment_bootstrap``.
"""

from __future__ import annotations

from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
import sys


def _load_bootstrap_module() -> ModuleType:
    """Load the canonical bootstrap implementation without recursive aliasing.

    In some source layouts this module can be imported as
    ``menace_sandbox.environment_bootstrap`` while still being used as a
    compatibility shim. Importing ``menace_sandbox.environment_bootstrap``
    directly from here would then recurse into this same file. To avoid that,
    load the implementation from the nested package path first.
    """

    module_candidates = (
        "menace_sandbox.menace_sandbox.environment_bootstrap",
        "menace_sandbox.environment_bootstrap",
    )
    for module_name in module_candidates:
        try:
            return import_module(module_name)
        except ImportError:
            continue

    impl_path = Path(__file__).resolve().with_name("menace_sandbox") / "environment_bootstrap.py"
    spec = spec_from_file_location("_menace_sandbox_environment_bootstrap_impl", impl_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load bootstrap module spec from {impl_path}")

    module = module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


_bootstrap_impl = _load_bootstrap_module()
BootstrapOrchestrator = _bootstrap_impl.BootstrapOrchestrator
EnvironmentBootstrapper = _bootstrap_impl.EnvironmentBootstrapper
ensure_bootstrapped = _bootstrap_impl.ensure_bootstrapped
ensure_bootstrapped_async = _bootstrap_impl.ensure_bootstrapped_async

# Keep exports explicit to avoid wildcard/partial-import drift in the shim API.

if EnvironmentBootstrapper is None:
    raise ImportError(
        "Failed to import EnvironmentBootstrapper from "
        "menace_sandbox.environment_bootstrap"
    )

__all__ = [
    "EnvironmentBootstrapper",
    "BootstrapOrchestrator",
    "ensure_bootstrapped",
    "ensure_bootstrapped_async",
]
