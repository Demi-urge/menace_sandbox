"""Lightweight bootstrap hook for the optional quick-fix background worker.

The production environment runs a dedicated service that continuously monitors
error reports and invokes :mod:`quick_fix_engine` to suggest corrective patches.
For local development and CI the full dependency chain is often unavailable,
causing the sandbox bootstrapper to warn that the service is missing.  The
helper in this repository keeps the bootstrap workflow quiet by providing a
minimal shim that can be safely imported in constrained environments.

The module purposely avoids importing heavy quick-fix dependencies at import
time.  When :func:`start` is invoked we attempt to import the real
:mod:`quick_fix_engine` package.  If the import succeeds a short log message is
emitted to confirm that the real service should be started separately.  When the
module is unavailable a debug log explains that the background worker is
skipped.  In either case the bootstrapper considers the optional service
"installed" which prevents noisy warnings during ``manual_bootstrap`` runs.
"""

from __future__ import annotations

import logging

__all__ = ["start", "is_running"]

_logger = logging.getLogger(__name__)
_started: bool = False


def start() -> None:
    """Best-effort bootstrap hook used by :mod:`sandbox_runner.bootstrap`.

    The real quick-fix worker is highly environment dependent; replicating it in
    the public sandbox would provide limited value while increasing maintenance
    overhead.  The shim therefore provides the minimal surface area required by
    the bootstrap process.  The function is idempotent and simply records that
    the service has been "started" to avoid duplicate log messages.
    """

    global _started
    if _started:
        return

    try:
        import quick_fix_engine  # noqa: F401  # local import for optional dep
    except Exception as exc:  # pragma: no cover - import failures depend on env
        _logger.debug(
            "quick_fix_engine unavailable; skipping optional background worker: %s",
            exc,
        )
    else:
        _logger.info(
            "quick_fix_engine_service shim loaded; start the production worker "
            "separately if desired",
        )
    finally:
        _started = True


def is_running() -> bool:
    """Return ``True`` once :func:`start` has been invoked."""

    return _started
