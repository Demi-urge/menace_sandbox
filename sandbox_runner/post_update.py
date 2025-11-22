from __future__ import annotations

from pathlib import Path
from .logging_utils import get_logger

from .orphan_integration import integrate_and_graph_orphans

logger = get_logger(__name__)


def integrate_orphans(
    repo: Path,
    router=None,
    *,
    context_builder: "ContextBuilder",
) -> list[str]:
    """Integrate newly created orphan modules.

    This function now delegates to
    :func:`sandbox_runner.orphan_integration.integrate_and_graph_orphans`
    so that discovery, module inclusion and graph updates follow a single
    implementation. ``context_builder`` is forwarded to that helper.
    """

    try:
        _, tested, _, _, _ = integrate_and_graph_orphans(
            repo, logger=logger, router=router, context_builder=context_builder
        )
    except Exception:  # pragma: no cover - best effort
        logger.exception("orphan integration failed")
        return []
    return tested.get("added", [])
