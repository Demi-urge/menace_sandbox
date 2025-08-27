from __future__ import annotations

"""Shared helpers for integrating orphan modules into the sandbox workflow."""

from pathlib import Path
from typing import Iterable, Dict, List, TYPE_CHECKING, Tuple

from logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - heavy import for type checking only
    from roi_tracker import ROITracker


def integrate_orphans(
    repo: Path,
    modules: Iterable[str] | None = None,
    *,
    logger=None,
    router=None,
) -> Tuple["ROITracker" | None, Dict[str, List[str]], List[int], bool, bool]:
    """Discover and integrate orphan modules into existing workflows.

    Parameters
    ----------
    repo:
        Root directory of the repository.
    modules:
        Optional iterable of repository-relative module paths to integrate. When
        omitted, :func:`sandbox_runner.orphan_discovery.discover_recursive_orphans`
        is used to locate candidates automatically.
    logger:
        Optional logger instance used for diagnostics.
    router:
        Optional database router forwarded to helpers.

    Returns
    -------
    tuple
        ``(tracker, results, updated_workflows, synergy_ok, cluster_ok)`` where
        ``results`` mirrors the mapping returned by
        :func:`sandbox_runner.environment.auto_include_modules`.
    """

    log = logger or get_logger(__name__)

    paths: List[str]
    if modules is None:
        try:
            from .orphan_discovery import discover_recursive_orphans

            mapping = discover_recursive_orphans(str(repo))
        except Exception:  # pragma: no cover - best effort
            log.exception("discover_recursive_orphans failed")
            return None, {}, [], False, False
        if not mapping:
            return None, {}, [], False, False
        paths = [
            Path(name.replace(".", "/")).with_suffix(".py").as_posix()
            for name in mapping
        ]
    else:
        paths = list(modules)
        if not paths:
            return None, {}, [], False, False

    try:
        from .environment import auto_include_modules, try_integrate_into_workflows
    except Exception:  # pragma: no cover - best effort
        log.exception("environment helpers import failed")
        return None, {}, [], False, False

    try:
        tracker, tested = auto_include_modules(paths, recursive=True, router=router)
    except Exception:  # pragma: no cover - best effort
        log.exception("auto include of discovered orphans failed")
        return None, {}, [], False, False

    added = tested.get("added", [])
    synergy_ok = False
    cluster_ok = False
    if added:
        try:
            from module_synergy_grapher import ModuleSynergyGrapher, load_graph

            grapher = ModuleSynergyGrapher(root=repo)
            graph_path = repo / "sandbox_data" / "module_synergy_graph.json"
            if graph_path.exists():
                try:
                    grapher.graph = load_graph(graph_path)
                except Exception:
                    grapher.graph = None
            if getattr(grapher, "graph", None) is None:
                try:
                    grapher.graph = grapher.build_graph(repo)
                except Exception:
                    grapher.graph = None
            if getattr(grapher, "graph", None) is not None:
                names = [Path(m).with_suffix("").as_posix() for m in added]
                grapher.update_graph(names)
                synergy_ok = True
        except Exception:  # pragma: no cover - best effort
            log.warning("module synergy update failed", exc_info=True)

        try:
            from intent_clusterer import IntentClusterer

            clusterer = IntentClusterer()
            clusterer.index_modules([repo / m for m in added])
            cluster_ok = True
        except Exception:  # pragma: no cover - best effort
            log.warning("intent clustering update failed", exc_info=True)

        try:
            updated = try_integrate_into_workflows(sorted(added), router=router) or []
        except Exception:  # pragma: no cover - best effort
            log.warning("workflow integration failed", exc_info=True)
            updated = []
    else:
        updated = []

    return tracker, tested, updated, synergy_ok, cluster_ok
