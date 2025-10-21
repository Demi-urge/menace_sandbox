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

print("🧪 ENTERED: quick_fix_engine_service.py top-level")

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
    print("[quick_fix_engine_service] start() invoked")
    if _started:
        print("[quick_fix_engine_service] start() called but already started; returning early")
        return

    try:
        print("[quick_fix_engine_service] attempting to import quick_fix_engine")
        import quick_fix_engine  # noqa: F401  # local import for optional dep
    except Exception as exc:  # pragma: no cover - import failures depend on env
        print(
            "[quick_fix_engine_service] quick_fix_engine import failed; skipping optional background worker",
            exc,
        )
        _logger.debug(
            "quick_fix_engine unavailable; skipping optional background worker: %s",
            exc,
        )
    else:
        print(
            "[quick_fix_engine_service] quick_fix_engine import succeeded; see logs for next steps"
        )
        _logger.info(
            "quick_fix_engine_service shim loaded; start the production worker "
            "separately if desired",
        )
    finally:
        print("[quick_fix_engine_service] marking service as started")
        _started = True


def is_running() -> bool:
    """Return ``True`` once :func:`start` has been invoked."""

    return _started
