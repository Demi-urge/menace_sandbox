from __future__ import annotations

from pathlib import Path
from logging_utils import get_logger

from .orphan_integration import integrate_and_graph_orphans

logger = get_logger(__name__)


def integrate_orphans(repo: Path, router=None) -> list[str]:
    """Integrate newly created orphan modules.

    This function now delegates to
    :func:`sandbox_runner.orphan_integration.integrate_and_graph_orphans`
    so that discovery, module inclusion and graph updates follow a single
    implementation.
    """

    try:
        _, tested, _, _, _ = integrate_and_graph_orphans(
            repo, logger=logger, router=router
        )
    except Exception:  # pragma: no cover - best effort
        logger.exception("orphan integration failed")
        return []
    return tested.get("added", [])
