"""Public bootstrap API for the :mod:`menace` compatibility package.

This module intentionally re-exports the package bootstrap entry points from
``menace_sandbox.environment_bootstrap`` so callers can reliably import
``menace.environment_bootstrap`` in both source and installed layouts.
"""

from menace_sandbox.environment_bootstrap import (
    BootstrapOrchestrator,
    EnvironmentBootstrapper,
    ensure_bootstrapped,
    ensure_bootstrapped_async,
)

__all__ = [
    "EnvironmentBootstrapper",
    "BootstrapOrchestrator",
    "ensure_bootstrapped",
    "ensure_bootstrapped_async",
]
