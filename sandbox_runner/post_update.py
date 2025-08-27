from __future__ import annotations

from pathlib import Path
from logging_utils import get_logger

logger = get_logger(__name__)


def integrate_orphans(repo: Path, router=None) -> list[str]:
    """Integrate newly created orphan modules.

    Parameters
    ----------
    repo:
        Root directory of the repository.
    router:
        Optional database router forwarded to helpers.

    Returns
    -------
    list[str]
        Repository-relative paths of modules added to the module map.
    """

    try:
        from .orphan_discovery import discover_recursive_orphans
    except Exception:  # pragma: no cover - best effort
        logger.exception("discover_recursive_orphans import failed")
        return []

    try:
        mapping = discover_recursive_orphans(str(repo))
    except Exception:  # pragma: no cover - best effort
        logger.exception("discover_recursive_orphans failed")
        return []

    if not mapping:
        return []

    try:
        from .environment import auto_include_modules, try_integrate_into_workflows
    except Exception:  # pragma: no cover - best effort
        logger.exception("environment helpers import failed")
        return []

    paths = [
        Path(name.replace(".", "/")).with_suffix(".py").as_posix()
        for name in mapping
    ]

    added: list[str] = []
    try:
        _, tested = auto_include_modules(paths, recursive=True, router=router)
        added = tested.get("added", [])
    except Exception:  # pragma: no cover - best effort
        logger.exception("auto include of discovered orphans failed")
        return []

    if not added:
        return []

    try:
        from module_synergy_grapher import ModuleSynergyGrapher, load_graph

        grapher = ModuleSynergyGrapher(root=repo)
        graph_path = repo / "sandbox_data" / "module_synergy_graph.json"
        if getattr(grapher, "graph", None) is None:
            try:
                if graph_path.exists():
                    grapher.graph = load_graph(graph_path)
                else:
                    grapher.graph = grapher.build_graph(repo)
            except Exception:  # pragma: no cover - best effort
                grapher.graph = None
        if getattr(grapher, "graph", None) is not None:
            names = [Path(m).with_suffix("").as_posix() for m in added]
            grapher.update_graph(names)
    except Exception:  # pragma: no cover - best effort
        logger.warning("module synergy update failed", exc_info=True)

    try:
        from intent_clusterer import IntentClusterer

        clusterer = IntentClusterer()
        clusterer.index_modules([repo / m for m in added])
    except Exception:  # pragma: no cover - best effort
        logger.warning("intent clustering update failed", exc_info=True)

    try:
        try_integrate_into_workflows(sorted(added), router=router)
    except Exception:  # pragma: no cover - best effort
        logger.warning("workflow integration failed", exc_info=True)

    return added
