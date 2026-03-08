"""Legacy compatibility wrapper for bootstrap imports.

Prefer importing from ``menace.environment_bootstrap``. This root-level module
is retained for external callers that still import ``environment_bootstrap``.
"""

from menace_sandbox.environment_bootstrap import (
    BootstrapOrchestrator,
    EnvironmentBootstrapper,
    ensure_bootstrapped,
    ensure_bootstrapped_async,
)

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
