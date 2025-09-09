from __future__ import annotations

"""Shared helpers for integrating orphan modules into the sandbox workflow."""

from pathlib import Path
from typing import Iterable, Dict, List, TYPE_CHECKING, Tuple

import json
from datetime import datetime

import yaml

from logging_utils import get_logger
from dynamic_path_router import resolve_path, resolve_module_path

if TYPE_CHECKING:  # pragma: no cover - heavy import for type checking only
    from roi_tracker import ROITracker


def integrate_and_graph_orphans(
    repo: Path,
    modules: Iterable[str] | None = None,
    *,
    logger=None,
    router=None,
    context_builder: "ContextBuilder",
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
    context_builder:
        :class:`~vector_service.context_builder.ContextBuilder` used for sandbox
        interactions.

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
        paths = [resolve_module_path(name).as_posix() for name in mapping]
    else:
        paths = [resolve_path(m).as_posix() for m in modules]
        if not paths:
            return None, {}, [], False, False

    try:
        from .environment import auto_include_modules, try_integrate_into_workflows
    except Exception:  # pragma: no cover - best effort
        log.exception("environment helpers import failed")
        return None, {}, [], False, False

    builder = context_builder
    try:
        tracker, tested = auto_include_modules(
            paths, recursive=True, router=router, context_builder=builder
        )
    except Exception:  # pragma: no cover - best effort
        log.exception("auto include of discovered orphans failed")
        return None, {}, [], False, False

    added = tested.get("added", [])
    synergy_ok = False
    cluster_ok = False
    workflow_ok = False
    if added:
        try:
            from module_synergy_grapher import ModuleSynergyGrapher, load_graph

            grapher = ModuleSynergyGrapher(root=repo)
            graph_path = resolve_path("sandbox_data/module_synergy_graph.json")
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
                names = [
                    Path(m).with_suffix("").relative_to(repo).as_posix()
                    for m in added
                ]
                grapher.update_graph(names)
                synergy_ok = True
        except Exception:  # pragma: no cover - best effort
            log.warning("module synergy update failed", exc_info=True)

        try:
            from intent_clusterer import IntentClusterer

            clusterer = IntentClusterer()
            clusterer.index_modules([resolve_path(m) for m in added])
            cluster_ok = True
        except Exception:  # pragma: no cover - best effort
            log.warning("intent clustering update failed", exc_info=True)

        try:
            updated = (
                try_integrate_into_workflows(
                    sorted(added), router=router, context_builder=builder
                )
                or []
            )
            workflow_ok = True
        except Exception:  # pragma: no cover - best effort
            log.warning("workflow integration failed", exc_info=True)
            updated = []
            workflow_ok = False
    else:
        updated = []
        workflow_ok = True

    retry: List[str] = []
    if added and not (synergy_ok and cluster_ok and workflow_ok):
        retry = list(added)
        tested["retry"] = retry

    log_path = resolve_path("sandbox_data/orphan_integration.log")
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "modules": added,
        "updated_workflows": updated,
        "synergy_ok": synergy_ok,
        "cluster_ok": cluster_ok,
        "workflow_ok": workflow_ok,
        "retry": retry,
    }
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception:  # pragma: no cover - best effort
        log.warning("failed to record orphan integration log", exc_info=True)

    return tracker, tested, updated, synergy_ok, cluster_ok


def _record_orphan_metrics(
    repo: Path, count: int, syn_ok: bool, cl_ok: bool, log
) -> None:
    """Persist orphan integration metrics to ``sandbox_metrics.yaml``.

    Parameters
    ----------
    repo:
        Repository root.
    count:
        Number of newly added modules.
    syn_ok:
        Whether the module synergy graph update succeeded.
    cl_ok:
        Whether intent clustering updated successfully.
    log:
        Logger used for diagnostic messages.
    """

    path = resolve_path("sandbox_metrics.yaml")
    data: Dict[str, Dict[str, float]] = {}
    try:
        if path.exists():
            data = yaml.safe_load(path.read_text()) or {}
    except Exception:  # pragma: no cover - best effort
        log.warning("failed to load sandbox metrics", exc_info=True)
        data = {}

    extra = data.setdefault("extra_metrics", {})
    extra["orphan_modules_added"] = float(count)
    extra["synergy_update_success"] = 1.0 if syn_ok else 0.0
    extra["intent_update_success"] = 1.0 if cl_ok else 0.0

    try:
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    except Exception:  # pragma: no cover - best effort
        log.warning("failed to record sandbox metrics", exc_info=True)


def post_round_orphan_scan(
    repo: Path,
    modules: Iterable[str] | None = None,
    *,
    logger=None,
    router=None,
    context_builder: "ContextBuilder",
) -> Tuple[List[str], bool, bool]:
    """Integrate orphan modules discovered after a round of code changes.

    This helper wraps :func:`integrate_and_graph_orphans` to perform
    recursive discovery via :func:`discover_recursive_orphans`, include any
    modules using :func:`auto_include_modules`, update the module synergy graph
    and intent clustering, and finally return the list of newly added module
    paths along with flags indicating whether the synergy graph and intent
    cluster updates succeeded. ``context_builder`` is forwarded to the helper
    to maintain consistent sandbox context.
    """

    log = logger or get_logger(__name__)

    _tracker, tested, _updated, syn_ok, cl_ok = integrate_and_graph_orphans(
        repo, modules, logger=log, router=router, context_builder=context_builder
    )
    added = tested.get("added", [])
    count = len(added)

    log.info(
        "post_round_orphan_scan added=%d synergy_ok=%s intent_ok=%s",
        count,
        syn_ok,
        cl_ok,
    )

    _record_orphan_metrics(repo, count, syn_ok, cl_ok, log)

    return added, syn_ok, cl_ok


# Backwards compatibility -------------------------------------------------
# Historically this utility was exported as ``integrate_orphans``.  Preserve
# the old name so existing callers continue to function while new code uses the
# more descriptive ``integrate_and_graph_orphans``.
integrate_orphans = integrate_and_graph_orphans
