"""Public bootstrap API for the :mod:`menace` compatibility package.

This module intentionally re-exports the package bootstrap entry points from
``menace_sandbox.environment_bootstrap`` so callers can reliably import
``menace.environment_bootstrap`` in both source and installed layouts.
"""

from menace_sandbox.environment_bootstrap import (
    BootstrapOrchestrator,
    EnvironmentBootstrapper,
    bootstrap_in_progress,
    ensure_bootstrapped,
    ensure_bootstrapped_async,
    is_bootstrapped,
)

__all__ = [
    "EnvironmentBootstrapper",
    "BootstrapOrchestrator",
    "is_bootstrapped",
    "bootstrap_in_progress",
    "ensure_bootstrapped",
    "ensure_bootstrapped_async",
]
