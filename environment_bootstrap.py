"""Compatibility wrapper for package-based bootstrap utilities."""

from menace_sandbox.environment_bootstrap import (
    BootstrapOrchestrator,
    EnvironmentBootstrapper,
    ensure_bootstrapped,
    ensure_bootstrapped_async,
)

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
