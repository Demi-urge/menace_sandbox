"""Entry point for launching the autonomous sandbox.

This wrapper bootstraps the environment and model paths automatically before
starting the sandbox. It captures startup exceptions and allows the log level
to be configured via ``SandboxSettings`` or overridden on the command line
without requiring any manual post-launch edits.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

if "--health-check" in sys.argv[1:]:
    if not os.getenv("SANDBOX_DEPENDENCY_MODE"):
        os.environ["SANDBOX_DEPENDENCY_MODE"] = "minimal"
    # Disable long-running monitoring loops during the lightweight health
    # probe so the command terminates promptly even when background services
    # would normally bootstrap DataBot.
    os.environ.setdefault("MENACE_SANDBOX_MODE", "health_check")
    os.environ.setdefault("MENACE_DISABLE_MONITORING", "1")

from logging_utils import get_logger, setup_logging, set_correlation_id, log_record
from sandbox_settings import SandboxSettings
from dependency_health import DependencyMode, resolve_dependency_mode
from sandbox.preseed_bootstrap import initialize_bootstrap_context
from sandbox_runner.bootstrap import (
    auto_configure_env,
    bootstrap_environment,
    launch_sandbox,
    sandbox_health,
    shutdown_autonomous_sandbox,
)
try:  # pragma: no cover - allow package relative import
    from metrics_exporter import (
        sandbox_restart_total,
        sandbox_crashes_total,
    sandbox_last_failure_ts,
)
except Exception:  # pragma: no cover - fallback when run as a module
    from .metrics_exporter import (  # type: ignore
        sandbox_restart_total,
        sandbox_crashes_total,
        sandbox_last_failure_ts,
    )

from dynamic_path_router import resolve_path
from shared_event_bus import event_bus as shared_event_bus
from workflow_evolution_manager import WorkflowEvolutionManager
import workflow_graph
from self_improvement.workflow_discovery import discover_workflow_specs
from self_improvement.component_workflow_synthesis import discover_component_workflows
from sandbox_orchestrator import SandboxOrchestrator
from context_builder_util import create_context_builder
from self_improvement.orphan_handling import integrate_orphans, post_round_orphan_scan
from self_improvement.meta_planning import (
    record_workflow_iteration,
    workflow_controller_status,
)

try:  # pragma: no cover - optional dependency
    from task_handoff_bot import WorkflowDB  # type: ignore
except Exception:  # pragma: no cover - allow sandbox startup without WorkflowDB
    WorkflowDB = None  # type: ignore

LOGGER = logging.getLogger(__name__)


def _resolve_dependency_mode(settings: SandboxSettings) -> DependencyMode:
    """Resolve the effective dependency handling policy for *settings*."""

    configured: str | None = getattr(settings, "dependency_mode", None)
    return resolve_dependency_mode(configured)


def _dependency_failure_messages(
    dependency_health: Mapping[str, Any] | None,
    *,
    dependency_mode: DependencyMode,
) -> list[str]:
    """Return user-facing failure reasons derived from dependency metadata."""

    if not isinstance(dependency_health, Mapping):
        return []

    missing: Sequence[Mapping[str, Any]] = tuple(
        item
        for item in dependency_health.get("missing", [])
        if isinstance(item, Mapping)
    )

    if not missing:
        return []

    required = [item for item in missing if not item.get("optional", False)]
    optional = [item for item in missing if item.get("optional", False)]

    failures: list[str] = []
    if required:
        failures.append(
            "missing required dependencies: "
            + ", ".join(sorted(str(item.get("name", "unknown")) for item in required))
        )
    if dependency_mode is not DependencyMode.MINIMAL and optional:
        failures.append(
            "missing optional dependencies in strict mode: "
            + ", ".join(sorted(str(item.get("name", "unknown")) for item in optional))
        )
    return failures


def _evaluate_health(
    health: Mapping[str, Any],
    *,
    dependency_mode: DependencyMode,
) -> tuple[bool, list[str]]:
    """Determine whether ``health`` represents a successful health check."""

    failures: list[str] = []

    if not health.get("databases_accessible", True):
        db_errors = health.get("database_errors")
        if isinstance(db_errors, Mapping) and db_errors:
            details = ", ".join(
                f"{name}: {error}"
                for name, error in sorted(db_errors.items())
            )
            failures.append(f"databases inaccessible ({details})")
        else:
            failures.append("databases inaccessible")

    failures.extend(
        _dependency_failure_messages(
            health.get("dependency_health"),
            dependency_mode=dependency_mode,
        )
    )

    return not failures, failures


def _emit_health_report(
    health: Mapping[str, Any],
    *,
    healthy: bool,
    failures: Sequence[str],
) -> None:
    """Write a structured health report to standard output."""

    payload = {
        "status": "pass" if healthy else "fail",
        "failures": list(failures),
        "health": health,
    }
    sys.stdout.write(json.dumps(payload, sort_keys=True))
    sys.stdout.write("\n")
    sys.stdout.flush()


def _load_workflow_records(
    settings: SandboxSettings,
    *,
    discovered_specs: Sequence[Mapping[str, Any]] | None = None,
) -> list[Mapping[str, Any]]:
    """Load workflow specs from configuration and the WorkflowDB if available."""

    records: list[Mapping[str, Any]] = []

    configured = getattr(settings, "workflow_specs", None) or getattr(
        settings, "meta_workflow_specs", None
    )
    if isinstance(configured, Sequence) and not isinstance(configured, (str, bytes)):
        records.extend([spec for spec in configured if isinstance(spec, Mapping)])

    if discovered_specs:
        records.extend([spec for spec in discovered_specs if isinstance(spec, Mapping)])

    if WorkflowDB is not None:
        try:
            wf_db = WorkflowDB(Path(settings.workflows_db))
            records.extend(wf_db.fetch_workflows(limit=200))
        except Exception:  # pragma: no cover - best effort hydration
            LOGGER.exception("failed to hydrate workflow records from WorkflowDB")

    return records


def _discover_repo_workflows(
    *, logger: logging.Logger, base_path: str | Path | None = None
) -> list[Mapping[str, Any]]:
    """Best-effort discovery of workflow-like modules and bots.

    This routine inspects the repository for ``workflow_*.py`` modules as well as
    bot modules ending in ``_bot.py``. Each finding is converted into a minimal
    workflow specification that downstream loaders can hydrate without manual
    configuration.
    """

    specs: list[Mapping[str, Any]] = []
    root = Path(base_path or resolve_path("."))

    try:
        specs.extend(discover_workflow_specs(base_path=root, logger=logger))
    except Exception:
        logger.exception(
            "workflow module discovery failed", extra=log_record(event="workflow-scan")
        )

    try:
        from bot_discovery import _iter_bot_modules

        for mod_path in _iter_bot_modules(root):
            module_name = ".".join(mod_path.relative_to(root).with_suffix("").parts)
            specs.append(
                {
                    "workflow": [module_name],
                    "workflow_id": module_name,
                    "task_sequence": [module_name],
                    "source": "bot_discovery",
                }
            )
    except Exception:
        logger.exception(
            "bot workflow discovery failed", extra=log_record(event="bot-scan")
        )

    return specs


def _decompose_menace_components(
    *,
    settings: SandboxSettings,
    logger: logging.Logger,
    workflow_evolver: WorkflowEvolutionManager,
) -> tuple[list[Mapping[str, Any]], dict[str, Callable[[], Any]]]:
    """Derive workflow specs and callables from the Menace monolith.

    The decomposition step inspects workflow-oriented modules in the repo and
    converts them into executable callables via the ``WorkflowEvolutionManager``
    so the meta-planning loop can mutate and wire them dynamically instead of
    depending solely on pre-seeded bootstrap entries.
    """

    repo_root = getattr(settings, "repo_root", None) or resolve_path(".")
    derived_specs: list[Mapping[str, Any]] = []
    derived_callables: dict[str, Callable[[], Any]] = {}

    try:
        derived_specs.extend(
            discover_workflow_specs(base_path=repo_root, logger=logger)
        )
    except Exception:
        logger.exception(
            "failed to decompose repo workflows",
            extra=log_record(event="menace-decomposition"),
        )

    for spec in derived_specs:
        seq = spec.get("workflow") or spec.get("task_sequence") or []
        workflow_id = str(
            spec.get("workflow_id")
            or spec.get("metadata", {}).get("workflow_id")
            or ""
        ).strip()
        if not workflow_id or not seq:
            continue
        try:
            seq_list = (
                list(seq)
                if isinstance(seq, Sequence) and not isinstance(seq, (str, bytes))
                else [seq]
            )
            derived_callables[workflow_id] = workflow_evolver.build_callable(
                "-".join(str(step) for step in seq_list)
            )
        except Exception:
            logger.exception(
                "failed to build callable for decomposed workflow",
                extra=log_record(workflow_id=workflow_id),
            )

    return derived_specs, derived_callables


def _build_self_improvement_workflows(
    bootstrap_context: Mapping[str, Any],
    settings: SandboxSettings,
    workflow_evolver: WorkflowEvolutionManager,
    *,
    logger: logging.Logger,
    discovered_specs: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Callable[[], Any]], workflow_graph.WorkflowGraph]:
    """Return workflow callables and relationship graph for meta planning."""

    workflows: dict[str, Callable[[], Any]] = {}
    graph = workflow_graph.WorkflowGraph()

    def _tag_node(workflow_id: str, **metadata: Any) -> None:
        try:
            if getattr(workflow_graph, "_HAS_NX", False):
                if graph.graph.has_node(workflow_id):
                    graph.graph.nodes[workflow_id].update(metadata)
            else:
                nodes = graph.graph.setdefault("nodes", {})
                nodes.setdefault(workflow_id, {}).update(metadata)
        except Exception:
            logger.debug(
                "failed to tag workflow node", extra=log_record(workflow_id=workflow_id)
            )

    all_discovered = list(discovered_specs or [])
    repo_root = getattr(settings, "repo_root", None)
    all_discovered.extend(
        _discover_repo_workflows(logger=logger, base_path=repo_root)
    )

    try:
        component_specs = discover_component_workflows(
            base_path=repo_root,
            logger=logger,
        )
        all_discovered.extend(component_specs)
    except Exception:
        logger.exception(
            "component workflow synthesis failed",
            extra=log_record(event="component-synthesis-error"),
        )

    derived_specs, derived_callables = _decompose_menace_components(
        settings=settings, logger=logger, workflow_evolver=workflow_evolver
    )
    all_discovered.extend(derived_specs)

    for name in (
        "manager",
        "pipeline",
        "engine",
        "registry",
        "data_bot",
        "context_builder",
    ):
        value = bootstrap_context.get(name)
        if value is None:
            continue

        workflow_id = f"preseeded_{name}"
        workflows[workflow_id] = (lambda v=value: v)
        graph.add_workflow(workflow_id)
        _tag_node(workflow_id, source="bootstrap", order=0)

    records = _load_workflow_records(settings, discovered_specs=all_discovered)
    if derived_callables:
        workflows.update(derived_callables)
        for wf_id in derived_callables:
            graph.add_workflow(wf_id)
            _tag_node(wf_id, source="monolith-decomposition", order=1)

    for record in records:
        seq = record.get("workflow") or record.get("task_sequence") or []
        workflow_id = str(
            record.get("id")
            or record.get("wid")
            or record.get("workflow_id")
            or record.get("name")
            or ""
        ).strip()
        if not workflow_id:
            continue

        try:
            seq_list = (
                list(seq)
                if isinstance(seq, Sequence) and not isinstance(seq, (str, bytes))
                else [seq]
            )
            seq_str = "-".join(str(step) for step in seq_list)
            workflows[workflow_id] = workflow_evolver.build_callable(seq_str)
            graph.add_workflow(workflow_id)
            _tag_node(
                workflow_id,
                source=record.get("source", "record"),
                order=len(seq_list),
            )
            if len(seq_list) > 1:
                for order, (src, dst) in enumerate(zip(seq_list, seq_list[1:]), start=1):
                    try:
                        graph.add_dependency(
                            str(src),
                            str(dst),
                            dependency_type="sequence",
                            order=order,
                        )
                    except Exception:
                        logger.debug(
                            "failed to wire workflow dependency",
                            extra=log_record(src=src, dst=dst, workflow_id=workflow_id),
                        )
        except Exception:  # pragma: no cover - defensive hydration
            logger.exception(
                "failed to hydrate workflow callable",
                extra=log_record(workflow_id=workflow_id),
            )

    context_builder = bootstrap_context.get("context_builder")
    if context_builder and hasattr(context_builder, "refresh_db_weights"):
        workflows.setdefault(
            "refresh_context", lambda cb=context_builder: cb.refresh_db_weights()
        )
        graph.add_workflow("refresh_context")
        _tag_node("refresh_context", source="bootstrap", order=0)

    include_orphans = bool(
        getattr(settings, "include_orphans", False)
        and not getattr(settings, "disable_orphans", False)
    )
    recursive_orphans = bool(getattr(settings, "recursive_orphan_scan", False))
    if include_orphans:
        workflows.setdefault(
            "integrate_orphans",
            lambda recursive=recursive_orphans: integrate_orphans(recursive=recursive),
        )
        graph.add_workflow("integrate_orphans")
        _tag_node("integrate_orphans", source="orphan_handling", order=0)

    if include_orphans and recursive_orphans:
        workflows.setdefault(
            "recursive_orphan_scan",
            lambda: post_round_orphan_scan(recursive=True),
        )
        graph.add_workflow("recursive_orphan_scan")
        _tag_node("recursive_orphan_scan", source="orphan_handling", order=1)

    logger.info(
        "registered %d workflows for meta planning",
        len(workflows),
        extra=log_record(workflow_count=len(workflows)),
    )

    return workflows, graph


def _roi_baseline_available() -> bool:
    """Return ``True`` when historical ROI signals exist on disk."""

    history_path = Path(
        os.environ.get(
            "WORKFLOW_ROI_HISTORY_PATH",
            resolve_path("workflow_roi_history.json"),
        )
    )
    return history_path.exists()


def _run_prelaunch_improvement_cycles(
    workflows: Mapping[str, Callable[[], Any]],
    planner_cls: type | None,
    settings: SandboxSettings,
    logger: logging.Logger,
    *,
    bootstrap_mode: bool = False,
) -> tuple[bool, bool]:
    """Iterate each workflow through ROI-gated improvement before launch."""

    logger.info(
        "✅ starting prelaunch ROI coordination",  # emoji for quick scanning
        extra=log_record(
            event="prelaunch-begin",
            workflow_count=len(workflows),
            planner_available=planner_cls is not None,
            bootstrap_mode=bootstrap_mode,
        ),
    )

    if not workflows:
        logger.error(
            "❌ no workflows available for ROI coordination; aborting sandbox launch",
            extra=log_record(event="meta-coordinator-missing-workflows"),
        )
        raise RuntimeError("no workflows available for ROI coordination")
    else:
        logger.info(
            "✅ workflows detected for ROI coordination",
            extra=log_record(
                event="meta-coordinator-workflows-present",
                workflow_count=len(workflows),
            ),
        )

    if planner_cls is None:
        logger.error(
            "❌ meta planner unavailable; cannot coordinate ROI stagnation",
            extra=log_record(event="meta-coordinator-missing"),
        )
        raise RuntimeError("meta planner unavailable for ROI coordination")
    else:
        logger.info(
            "✅ meta planner located for ROI coordination",
            extra=log_record(event="meta-coordinator-planner-present"),
        )

    system_ready = True
    roi_backoff = False
    per_workflow_ready: dict[str, bool] = {}

    for workflow_id, callable_fn in workflows.items():
        logger.info(
            "ℹ️ coordinating workflow for prelaunch ROI gate",
            extra=log_record(
                event="prelaunch-workflow-start",
                workflow_id=workflow_id,
                continuous_monitor=True,
            ),
        )
        ready, backoff = _coordinate_workflows_until_stagnation(
            {workflow_id: callable_fn},
            planner_cls=planner_cls,
            settings=settings,
            logger=logger,
            continuous_monitor=True,
            cycle_budget=3,
        )
        per_workflow_ready[workflow_id] = ready
        roi_backoff = roi_backoff or backoff
        logger.info(
            "✅ workflow ROI gate completed" if ready else "❌ workflow ROI gate incomplete",
            extra=log_record(
                event="prelaunch-workflow-result",
                workflow_id=workflow_id,
                ready=ready,
            roi_backoff=backoff,
        ),
    )
        if not ready:
            system_ready = False
            logger.warning(
                "❌ workflow stalled before launch",
                extra=log_record(workflow_id=workflow_id, event="prelaunch-stall"),
            )
        elif backoff:
            logger.warning(
                "❌ workflow hit ROI backoff during prelaunch",
                extra=log_record(
                    event="prelaunch-workflow-backoff",
                    workflow_id=workflow_id,
                    roi_backoff=True,
                ),
            )
        else:
            logger.info(
                "✅ workflow cleared ROI gate without backoff",
                extra=log_record(
                    event="prelaunch-workflow-clear",
                    workflow_id=workflow_id,
                    roi_backoff=False,
                ),
            )

    if system_ready and not roi_backoff:
        logger.info(
            "ℹ️ validating combined workflows for ROI stagnation",
            extra=log_record(
                event="prelaunch-system-check",
                workflow_count=len(workflows),
                continuous_monitor=True,
            ),
        )
        system_ready, system_backoff = _coordinate_workflows_until_stagnation(
            workflows,
            planner_cls=planner_cls,
            settings=settings,
            logger=logger,
            continuous_monitor=True,
        )
        roi_backoff = roi_backoff or system_backoff
        logger.info(
            "✅ combined ROI gate reached" if system_ready else "❌ combined ROI gate incomplete",
            extra=log_record(
                event="prelaunch-system-result",
                workflow_count=len(workflows),
                ready=system_ready,
                roi_backoff=system_backoff,
            ),
        )
    else:
        logger.info(
            "ℹ️ skipping combined ROI gate because a workflow stalled or backoff triggered",
            extra=log_record(
                event="prelaunch-system-skip",
                system_ready=system_ready,
                roi_backoff=roi_backoff,
            ),
        )

    snapshot = workflow_controller_status()
    if snapshot:
        logger.info(
            "ℹ️ workflow controller status snapshot",
            extra=log_record(event="prelaunch-controller-status", controllers=snapshot),
        )

    ready = system_ready and all(per_workflow_ready.values())

    logger.info(
        "✅ per-workflow ROI gates cleared" if ready else "❌ one or more workflows blocked",
        extra=log_record(
            event="prelaunch-per-workflow-status",
            ready=ready,
            roi_backoff=roi_backoff,
            workflow_count=len(per_workflow_ready),
            blocked_workflows=[k for k, v in per_workflow_ready.items() if not v],
        ),
    )

    if (
        bootstrap_mode
        and workflows
        and planner_cls is not None
        and not roi_backoff
        and not ready
    ):
        logger.info(
            "✅ bypassing diminishing returns gate during bootstrap; ROI baseline unavailable",
            extra=log_record(
                event="meta-coordinator-bootstrap-bypass",
                workflow_count=len(workflows),
            ),
        )
        ready = True
        logger.info(
            "✅ bootstrap bypass activated; launching despite missing ROI baseline",
            extra=log_record(event="meta-coordinator-bootstrap-bypass-applied"),
        )

    logger.info(
        "✅ prelaunch ROI coordination complete" if ready else "❌ prelaunch ROI coordination incomplete",
        extra=log_record(
            event="prelaunch-complete",
            ready=ready,
            roi_backoff=roi_backoff,
            bootstrap_mode=bootstrap_mode,
        ),
    )

    return ready, roi_backoff


def _coordinate_workflows_until_stagnation(
    workflows: Mapping[str, Callable[[], Any]],
    *,
    planner_cls: type | None,
    settings: SandboxSettings,
    logger: logging.Logger,
    continuous_monitor: bool = False,
    cycle_budget: int | None = None,
) -> tuple[bool, bool]:
    """Iterate workflows through the meta planner until ROI gains stagnate."""

    roi_settings = getattr(settings, "roi", None)
    threshold = float(
        getattr(roi_settings, "stagnation_threshold", 0.0)
        if roi_settings is not None
        else 0.0
    )
    streak_required = max(
        1,
        int(
            getattr(roi_settings, "stagnation_cycles", 1)
            if roi_settings is not None
            else 1
        ),
    )

    logger.info(
        "🔍 attempting to initialize meta planner for ROI coordination",
        extra=log_record(event="meta-coordinator-init-begin"),
    )
    try:
        planner = planner_cls(context_builder=create_context_builder())
        logger.info(
            "✅ meta planner initialized for ROI coordination",
            extra=log_record(event="meta-coordinator-init-success"),
        )
    except Exception:
        logger.exception(
            "❌ failed to initialize meta planner for ROI coordination",
            extra=log_record(event="meta-coordinator-init-error"),
        )
        raise

    for name, value in {
        "mutation_rate": settings.meta_mutation_rate,
        "roi_weight": settings.meta_roi_weight,
        "domain_transition_penalty": settings.meta_domain_penalty,
    }.items():
        if hasattr(planner, name):
            setattr(planner, name, value)
            logger.info(
                "✅ applied planner setting",  # emoji for quick scanning
                extra=log_record(
                    event="meta-coordinator-setting-applied",
                    setting=name,
                    value=value,
                ),
            )
        else:
            logger.debug(
                "ℹ️ planner setting skipped; attribute missing",
                extra=log_record(event="meta-coordinator-setting-skipped", setting=name),
            )

    diminishing: set[str] = set()
    roi_backoff_triggered = False
    budget = cycle_budget or max(len(workflows) * streak_required * 2, 3)

    for cycle in range(budget):
        try:
            records = planner.discover_and_persist(workflows)
        except Exception:
            logger.exception(
                "❌ meta planner coordination failed",
                extra=log_record(event="meta-coordinator-error", cycle=cycle),
            )
            break

        if not records:
            logger.info(
                "✅ meta planner returned no records; assuming diminishing returns",
                extra=log_record(event="meta-coordinator-empty", cycle=cycle),
            )
            break

        for rec in records:
            chain = rec.get("chain", [])
            chain_id = "->".join(chain) if chain else rec.get("workflow_id", "unknown")
            roi_gain = float(rec.get("roi_gain", 0.0))
            stats: dict[str, float] = {}
            if getattr(planner, "roi_db", None) is not None and isinstance(chain_id, str):
                try:
                    stats = planner.roi_db.fetch_chain_stats(chain_id)  # type: ignore[operator]
                except Exception:
                    logger.debug(
                        "roi stats lookup failed", extra=log_record(workflow_id=chain_id)
                    )

            roi_delta = float(stats.get("delta_roi", roi_gain))
            streak = int(stats.get("non_positive_streak", 0))
            controller_state = record_workflow_iteration(
                chain_id,
                roi_gain=roi_gain,
                roi_delta=roi_delta,
                threshold=threshold,
                patience=streak_required,
            )
            stagnated = roi_delta <= threshold and streak >= streak_required
            if controller_state.get("status") == "halted":
                stagnated = True
                logger.info(
                    "✅ workflow controller halted improvements",
                    extra=log_record(
                        workflow_id=chain_id,
                        roi_delta=controller_state.get("last_delta", roi_gain),
                        threshold=controller_state.get("threshold", threshold),
                        event="meta-controller-halt",
                    ),
                )

            logger.info(
                "meta planning progress",
                extra=log_record(
                    workflow_id=chain_id,
                    roi_gain=roi_gain,
                    roi_delta=roi_delta,
                    stagnation_threshold=threshold,
                    non_positive_streak=streak,
                    stagnation_met=stagnated,
                    controller_status=controller_state,
                    diminishing_complete=len(diminishing),
                    diminishing_target=len(workflows),
                    cycle=cycle,
                ),
            )

            if stagnated and isinstance(chain_id, str):
                diminishing.add(chain_id)

            if continuous_monitor and roi_delta <= threshold:
                roi_backoff_triggered = True
                logger.warning(
                    "roi backoff triggered during coordination",
                    extra=log_record(
                        workflow_id=chain_id,
                        roi_delta=roi_delta,
                        stagnation_threshold=threshold,
                        non_positive_streak=streak,
                        event="roi-backoff",
                    ),
                )
                break

        if roi_backoff_triggered or len(diminishing) >= len(workflows):
            break

    ready = len(diminishing) >= len(workflows)
    if not ready:
        logger.warning(
            "diminishing returns not reached for all workflows",
            extra=log_record(
                achieved=len(diminishing),
                total=len(workflows),
                event="meta-coordinator-incomplete",
            ),
        )

    return ready, roi_backoff_triggered


def main(argv: list[str] | None = None) -> None:
    """Launch the sandbox with optional log level configuration.

    Parameters
    ----------
    argv:
        Optional list of command line arguments. If ``None`` the arguments will
        be pulled from :data:`sys.argv`.
    """

    print("[start_autonomous_sandbox] main() entry", flush=True)

    argv_list = list(sys.argv[1:] if argv is None else argv)
    if "--health-check" in argv_list and not os.getenv("SANDBOX_DEPENDENCY_MODE"):
        os.environ["SANDBOX_DEPENDENCY_MODE"] = "minimal"

    settings = SandboxSettings()
    # Automatically configure the environment before proceeding so the caller
    # does not need to pre-populate configuration files or model paths.
    auto_configure_env(settings)
    # Reload settings to pick up any values written by ``auto_configure_env``.
    settings = SandboxSettings()

    parser = argparse.ArgumentParser(description="Launch the autonomous sandbox")
    parser.add_argument(
        "--log-level",
        dest="log_level",
        default=settings.sandbox_log_level,
        help="Logging level (e.g. DEBUG, INFO, WARNING)",
    )
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Run sandbox health checks and exit",
    )
    parser.add_argument(
        "--monitor-roi-backoff",
        action="store_true",
        help="Continuously monitor ROI backoff and pause launch when triggered",
    )
    parser.add_argument(
        "--include-orphans",
        dest="include_orphans",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include orphan modules during testing and planning",
    )
    parser.add_argument(
        "--recursive-orphan-scan",
        dest="recursive_orphan_scan",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable recursive orphan discovery and integration",
    )
    args = parser.parse_args(argv_list)

    if args.include_orphans is not None:
        os.environ["SANDBOX_INCLUDE_ORPHANS"] = "1" if args.include_orphans else "0"
        settings.include_orphans = bool(args.include_orphans)
    if args.recursive_orphan_scan is not None:
        os.environ["SANDBOX_RECURSIVE_ORPHANS"] = (
            "1" if args.recursive_orphan_scan else "0"
        )
        settings.recursive_orphan_scan = bool(args.recursive_orphan_scan)

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        setup_logging(level=args.log_level)
    else:
        root_logger.setLevel(getattr(logging, str(args.log_level).upper(), logging.INFO))
    cid = f"sas-{uuid.uuid4()}"
    set_correlation_id(cid)
    logger = get_logger(__name__)
    sandbox_restart_total.labels(service="start_autonomous", reason="launch").inc()
    logger.info("sandbox start", extra=log_record(event="start"))

    ready_to_launch = True
    roi_backoff_triggered = False

    try:
        if not args.health_check:
            try:
                bootstrap_context = initialize_bootstrap_context()
                os.environ.setdefault("META_PLANNING_LOOP", "1")
                os.environ.setdefault("META_PLANNING_INTERVAL", "10")
                os.environ.setdefault("META_IMPROVEMENT_THRESHOLD", "0.01")
                from self_improvement import meta_planning
                from self_improvement.meta_planning import (  # noqa: F401
                    self_improvement_cycle,
                )

                meta_planning.reload_settings(settings)
                workflow_evolver = WorkflowEvolutionManager()
                planner_cls = meta_planning.resolve_meta_workflow_planner(
                    force_reload=True
                )
                if planner_cls is None:
                    logger.error(
                        "MetaWorkflowPlanner not found; aborting sandbox launch",
                        extra=log_record(event="meta-planning-missing"),
                    )
                    sys.exit(1)

                interval = float(
                    os.getenv(
                        "META_PLANNING_INTERVAL",
                        getattr(settings, "meta_planning_interval", 10),
                    )
                )
                discovered_specs = []
                try:
                    discovered_specs = discover_workflow_specs(logger=logger)
                except Exception:
                    logger.exception(
                        "failed to auto-discover workflow specs",
                        extra=log_record(event="workflow-discovery-error"),
                    )

                orphan_specs: list[Mapping[str, Any]] = []
                include_orphans = bool(
                    getattr(settings, "include_orphans", False)
                    and not getattr(settings, "disable_orphans", False)
                )
                recursive_orphans = bool(
                    getattr(settings, "recursive_orphan_scan", False)
                )
                if include_orphans:
                    try:
                        orphan_modules = integrate_orphans(recursive=recursive_orphans)
                        orphan_specs.extend(
                            {
                                "workflow": [module],
                                "workflow_id": module,
                                "task_sequence": [module],
                                "source": "orphan_discovery",
                            }
                            for module in orphan_modules
                            if isinstance(module, str)
                        )
                    except Exception:
                        logger.exception(
                            "startup orphan integration failed",
                            extra=log_record(event="startup-orphan-discovery"),
                        )
                    if recursive_orphans:
                        try:
                            result = post_round_orphan_scan(recursive=True)
                            integrated = (
                                result.get("integrated")
                                if isinstance(result, Mapping)
                                else None
                            )
                            if integrated:
                                orphan_specs.extend(
                                    {
                                        "workflow": [module],
                                        "workflow_id": module,
                                        "task_sequence": [module],
                                        "source": "recursive_orphan_discovery",
                                    }
                                    for module in integrated
                                    if isinstance(module, str)
                                )
                        except Exception:
                            logger.exception(
                                "startup recursive orphan scan failed",
                                extra=log_record(event="startup-orphan-recursive"),
                            )

                workflows, workflow_graph_obj = _build_self_improvement_workflows(
                    bootstrap_context,
                    settings,
                    workflow_evolver,
                    logger=logger,
                    discovered_specs=[*discovered_specs, *orphan_specs],
                )
                if not workflows:
                    logger.error(
                        "no workflows discovered; startup halted before launching sandbox",
                        extra=log_record(
                            event="startup-no-workflows",
                            planner_available=planner_cls is not None,
                        ),
                    )
                    sys.exit(1)
                if planner_cls is None:
                    logger.error(
                        "planner resolution failed; cannot coordinate ROI for launch",
                        extra=log_record(event="startup-no-planner", workflow_count=len(workflows)),
                    )
                    sys.exit(1)
                bootstrap_mode = not _roi_baseline_available()
                logger.info(
                    "evaluating sandbox startup readiness",
                    extra=log_record(
                        event="startup-readiness",
                        workflow_count=len(workflows),
                        planner_available=planner_cls is not None,
                        bootstrap_mode=bootstrap_mode,
                    ),
                )
                readiness_error: str | None = None
                logger.info(
                    "🧭 meta-planning gate: beginning last-mile checks before launch",
                    extra=log_record(
                        event="meta-planning-gate-begin",
                        workflow_ids=list(workflows.keys()),
                        planner_resolved=planner_cls is not None,
                        bootstrap_mode=bootstrap_mode,
                    ),
                )
                try:
                    ready_to_launch, roi_backoff_triggered = _run_prelaunch_improvement_cycles(
                        workflows,
                        planner_cls=planner_cls,
                        settings=settings,
                        logger=logger,
                        bootstrap_mode=bootstrap_mode,
                    )
                    logger.info(
                        "✅ prelaunch ROI cycles finished without raising",  # emoji for quick scanning
                        extra=log_record(
                            event="startup-prelaunch-success",
                            ready_to_launch=ready_to_launch,
                            roi_backoff=roi_backoff_triggered,
                            workflow_count=len(workflows),
                            planner_available=planner_cls is not None,
                        ),
                    )
                except RuntimeError as exc:
                    readiness_error = str(exc)
                    ready_to_launch = False
                    roi_backoff_triggered = False
                    logger.error(
                        "❌ runtime error during prelaunch ROI cycles",  # emoji for quick scanning
                        extra=log_record(
                            event="startup-prelaunch-runtime-error",
                            readiness_error=readiness_error,
                            workflow_count=len(workflows),
                            planner_available=planner_cls is not None,
                        ),
                    )
                except Exception as exc:
                    readiness_error = f"unexpected prelaunch failure: {exc}"
                    ready_to_launch = False
                    roi_backoff_triggered = False
                    logger.exception(
                        "❌ unexpected exception during prelaunch ROI cycles",  # emoji for quick scanning
                        extra=log_record(
                            event="startup-prelaunch-unexpected-error",
                            readiness_error=readiness_error,
                            workflow_count=len(workflows),
                            planner_available=planner_cls is not None,
                        ),
                    )
                finally:
                    logger.info(
                        "ℹ️ prelaunch ROI cycle invocation finished",
                        extra=log_record(
                            event="startup-prelaunch-finished",
                            ready_to_launch=ready_to_launch,
                            roi_backoff=roi_backoff_triggered,
                            readiness_error=readiness_error,
                        ),
                    )

                meta_planning.reload_settings(settings)
                logger.info(
                    "startup readiness evaluation complete",
                    extra=log_record(
                        event="startup-readiness-result",
                        workflow_count=len(workflows),
                        ready_to_launch=ready_to_launch,
                        roi_backoff=roi_backoff_triggered,
                        planner_available=planner_cls is not None,
                    ),
                )
                logger.info(
                    "🔬 meta-planning readiness diagnostics collected",
                    extra=log_record(
                        event="meta-planning-readiness-diagnostics",
                        planner_status="✅ available" if planner_cls else "❌ missing",
                        workflow_status="✅ present" if workflows else "❌ none discovered",
                        roi_gate="✅ clear" if ready_to_launch else "❌ blocked",
                        roi_backoff="✅ none" if not roi_backoff_triggered else "❌ backoff",
                        readiness_error=readiness_error,
                        workflow_ids=list(workflows.keys()),
                    ),
                )
                logger.info(
                    "🧭 meta-planning gate: evaluating final decision criteria",
                    extra=log_record(
                        event="meta-planning-gate-eval",
                        has_workflows=bool(workflows),
                        planner_resolved=planner_cls is not None,
                        ready_to_launch=ready_to_launch,
                        roi_backoff=roi_backoff_triggered,
                        bootstrap_mode=bootstrap_mode,
                    ),
                )
                logger.info(
                    "✅ checkpoint: workflows present" if workflows else "❌ checkpoint failed: no workflows present",
                    extra=log_record(
                        event="meta-planning-gate-workflows",
                        condition_passed=bool(workflows),
                        workflow_count=len(workflows),
                    ),
                )
                logger.info(
                    "✅ checkpoint: planner resolved" if planner_cls is not None else "❌ checkpoint failed: planner missing",
                    extra=log_record(
                        event="meta-planning-gate-planner",
                        condition_passed=planner_cls is not None,
                        planner_cls=str(planner_cls),
                    ),
                )
                logger.info(
                    "✅ checkpoint: prelaunch ROI gate cleared" if ready_to_launch else "❌ checkpoint failed: ROI gate blocked",
                    extra=log_record(
                        event="meta-planning-gate-roi-ready",
                        condition_passed=ready_to_launch,
                        roi_backoff=roi_backoff_triggered,
                        readiness_error=readiness_error,
                    ),
                )
                logger.info(
                    "✅ checkpoint: no ROI backoff detected" if not roi_backoff_triggered else "❌ checkpoint failed: ROI backoff active",
                    extra=log_record(
                        event="meta-planning-gate-roi-backoff",
                        condition_passed=not roi_backoff_triggered,
                        roi_backoff=roi_backoff_triggered,
                    ),
                )
                logger.info(
                    "🔎 meta-planning launch decision inputs gathered",
                    extra=log_record(
                        event="meta-planning-gate-inputs",
                        ready_to_launch=ready_to_launch,
                        planner_resolved=planner_cls is not None,
                        workflow_count=len(workflows),
                        bootstrap_mode=bootstrap_mode,
                        roi_backoff=roi_backoff_triggered,
                        readiness_error=readiness_error,
                    ),
                )
                logger.info(
                    "🔍 evaluating meta planning launch gate",
                    extra=log_record(
                        event="meta-planning-launch-eval",
                        ready_to_launch=ready_to_launch,
                        roi_backoff=roi_backoff_triggered,
                        planner_resolved=planner_cls is not None,
                        workflow_count=len(workflows),
                        workflow_ids=list(workflows.keys()),
                        bootstrap_mode=bootstrap_mode,
                        readiness_error=readiness_error,
                    ),
                )
                logger.info(
                    "🔁 meta-planning gate status report (workflows=%s, planner=%s, backoff=%s, ready=%s)",
                    len(workflows),
                    bool(planner_cls),
                    roi_backoff_triggered,
                    ready_to_launch,
                    extra=log_record(
                        event="meta-planning-gate-status",
                        workflow_count=len(workflows),
                        planner_resolved=planner_cls is not None,
                        roi_backoff=roi_backoff_triggered,
                        ready_to_launch=ready_to_launch,
                        readiness_error=readiness_error,
                    ),
                )
                logger.info(
                    "🔎 meta planning gate checkpoints: workflows=%d, planner=%s, ready=%s, backoff=%s",
                    len(workflows),
                    bool(planner_cls),
                    ready_to_launch,
                    roi_backoff_triggered,
                    extra=log_record(
                        event="meta-planning-gate-checkpoints",
                        workflow_count=len(workflows),
                        planner_resolved=planner_cls is not None,
                        ready_to_launch=ready_to_launch,
                        roi_backoff=roi_backoff_triggered,
                        readiness_error=readiness_error,
                    ),
                )
                gating_checklist = {
                    "workflows_present": bool(workflows),
                    "planner_resolved": planner_cls is not None,
                    "roi_gate_clear": ready_to_launch,
                    "roi_backoff_clear": not roi_backoff_triggered,
                }
                for check, passed in gating_checklist.items():
                    logger.info(
                        "✅ gating checkpoint passed: %s" % check
                        if passed
                        else "❌ gating checkpoint failed: %s" % check,
                        extra=log_record(
                            event="meta-planning-gate-check", check=check, passed=passed
                        ),
                    )
                logger.info(
                    "🔦 meta-planning gate checklist compiled",
                    extra=log_record(event="meta-planning-gate-checklist", **gating_checklist),
                )
                logger.info(
                    "✅ meta planning gate decision computed; entering final launch guard",
                    extra=log_record(
                        event="meta-planning-gate-decision",
                        workflows_present=bool(workflows),
                        planner_available=planner_cls is not None,
                        ready_to_launch=ready_to_launch,
                        roi_backoff=roi_backoff_triggered,
                        readiness_error=readiness_error,
                        bootstrap_mode=bootstrap_mode,
                    ),
                )
                if not workflows:
                    logger.error(
                        "❌ gating halted: no workflows discovered for meta planning",
                        extra=log_record(event="meta-planning-gate-no-workflows"),
                    )
                if planner_cls is None:
                    logger.error(
                        "❌ gating halted: MetaWorkflowPlanner unresolved",
                        extra=log_record(event="meta-planning-gate-no-planner"),
                    )
                if roi_backoff_triggered:
                    logger.error(
                        "❌ gating halted: ROI backoff triggered before launch",
                        extra=log_record(event="meta-planning-gate-backoff"),
                    )
                if readiness_error:
                    logger.error(
                        "❌ gating halted: readiness error encountered",
                        extra=log_record(
                            event="meta-planning-gate-readiness-error",
                            readiness_error=readiness_error,
                        ),
                    )
                failure_reasons: list[str] = []
                if not workflows:
                    failure_reasons.append("no workflows discovered for meta planning")
                if planner_cls is None:
                    failure_reasons.append("MetaWorkflowPlanner unresolved")
                if roi_backoff_triggered:
                    failure_reasons.append("ROI backoff triggered before launch")
                if readiness_error:
                    failure_reasons.append(readiness_error)
                if not ready_to_launch and not readiness_error:
                    failure_reasons.append(
                        "workflows did not meet diminishing returns threshold"
                    )
                logger.info(
                    "🔦 meta-planning gate failure reasons compiled",
                    extra=log_record(
                        event="meta-planning-gate-failure-reasons",
                        failure_reasons=failure_reasons,
                        checklist=gating_checklist,
                    ),
                )
                logger.info(
                    "🧭 meta-planning gate summary computed; preparing branch selection",
                    extra=log_record(
                        event="meta-planning-gate-branch-summary",
                        ready_to_launch=ready_to_launch,
                        roi_backoff=roi_backoff_triggered,
                        planner_available=planner_cls is not None,
                        workflows_present=bool(workflows),
                        failure_reasons=failure_reasons,
                    ),
                )
                logger.info(
                    "🔎 launch condition breakdown: workflows=%s, planner=%s, roi_backoff=%s, readiness_error=%s",
                    bool(workflows),
                    bool(planner_cls),
                    roi_backoff_triggered,
                    readiness_error,
                    extra=log_record(
                        event="meta-planning-launch-breakdown",
                        has_workflows=bool(workflows),
                        planner_available=planner_cls is not None,
                        roi_backoff=roi_backoff_triggered,
                        readiness_error=readiness_error,
                    ),
                )
                if ready_to_launch:
                    logger.info(
                        "✅ gating green: all launch conditions satisfied; proceeding to thread bootstrap",
                        extra=log_record(
                            event="meta-planning-gate-green",
                            workflow_count=len(workflows),
                            planner_resolved=planner_cls is not None,
                            roi_backoff=roi_backoff_triggered,
                        ),
                    )
                    logger.info(
                        "🧭 meta planning start: entering bootstrap+thread block (expect subsequent checkpoints)",
                        extra=log_record(
                            event="meta-planning-start-block-enter",
                            workflow_ids=list(workflows.keys()),
                            planner_resolved=planner_cls is not None,
                            interval_seconds=interval,
                            bootstrap_mode=bootstrap_mode,
                            roi_backoff=roi_backoff_triggered,
                            ready_to_launch=ready_to_launch,
                        ),
                    )
                    logger.info(
                        "✅ prelaunch checks passed; proceeding with meta-planning start sequence",
                        extra=log_record(
                            event="meta-planning-launch-sequence-begin",
                            workflow_ids=list(workflows.keys()),
                            planner_class=str(planner_cls),
                            roi_backoff=roi_backoff_triggered,
                            bootstrap_mode=bootstrap_mode,
                        ),
                    )
                    logger.info(
                        "✅ launch gate green: ROI stagnation satisfied and planner resolved",
                        extra=log_record(
                            event="meta-planning-launch-green",
                            workflow_ids=list(workflows.keys()),
                            planner_cls=str(planner_cls),
                            roi_backoff=roi_backoff_triggered,
                        ),
                    )
                    logger.info(
                        "✅ readiness gate cleared; preparing to start meta planning loop",
                        extra=log_record(
                            event="meta-planning-ready",
                            roi_backoff=roi_backoff_triggered,
                            workflow_count=len(workflows),
                            planner_resolved=planner_cls is not None,
                            bootstrap_mode=bootstrap_mode,
                            prelaunch_ready=ready_to_launch,
                        ),
                    )
                    logger.info(
                        "✅ meta-planning launch prerequisites satisfied",
                        extra=log_record(
                            event="meta-planning-prereqs",
                            planner_cls=str(planner_cls),
                            interval_seconds=interval,
                            workflow_ids=list(workflows.keys()),
                            workflow_graph_built=workflow_graph_obj is not None,
                        ),
                    )
                    logger.info(
                        "✅ gating checklist satisfied; proceeding to meta planning bootstrap",
                        extra=log_record(
                            event="meta-planning-gate-green-checklist",
                            checklist=gating_checklist,
                            workflow_ids=list(workflows.keys()),
                        ),
                    )
                    logger.info(
                        "✅ meta planning gate satisfied; initializing launch choreography",
                        extra=log_record(
                            event="meta-planning-gate-satisfied",
                            workflow_ids=list(workflows.keys()),
                            planner_resolved=planner_cls is not None,
                            bootstrap_mode=bootstrap_mode,
                        ),
                    )
                    logger.info(
                        "🔧 configuring meta planning loop thread creation",
                        extra=log_record(
                            event="meta-planning-thread-config",
                            interval_seconds=interval,
                            workflow_graph_built=workflow_graph_obj is not None,
                            workflow_count=len(workflows),
                        ),
                    )
                    logger.info(
                        "🧠 preparing to invoke start_self_improvement_cycle() with event bus and workflow graph",
                        extra=log_record(
                            event="meta-planning-pre-bootstrap-call",
                            workflow_ids=list(workflows.keys()),
                            planner_cls=str(planner_cls),
                            interval_seconds=interval,
                            workflow_graph_present=workflow_graph_obj is not None,
                            event_bus_available=shared_event_bus is not None,
                            workflow_graph_nodes=
                                list(workflow_graph_obj.keys())
                                if isinstance(workflow_graph_obj, Mapping)
                                else None,
                        ),
                    )
                    try:
                        logger.info(
                            "🔧 invoking meta planning loop bootstrap",
                            extra=log_record(
                                event="meta-planning-bootstrap-call",
                                workflow_count=len(workflows),
                                planner_cls=str(planner_cls),
                                interval_seconds=interval,
                                workflow_graph_present=workflow_graph_obj is not None,
                                workflow_graph_len=len(workflow_graph_obj or {}),
                                event_bus_connected=shared_event_bus is not None,
                            ),
                        )
                        logger.info(
                            "🛰️ meta planning bootstrap call about to execute start_self_improvement_cycle()",
                            extra=log_record(
                                event="meta-planning-bootstrap-about-to-call",
                                workflow_count=len(workflows),
                                planner_cls=str(planner_cls),
                                interval_seconds=interval,
                                workflow_graph_keys=list((workflow_graph_obj or {}).keys())
                                if isinstance(workflow_graph_obj, Mapping)
                                else None,
                                workflow_graph_type=type(workflow_graph_obj).__name__,
                                event_bus_type=type(shared_event_bus).__name__
                                if shared_event_bus is not None
                                else None,
                            ),
                        )
                        logger.info(
                            "🛰️ verifying start_self_improvement_cycle callable availability before invoke",
                            extra=log_record(
                                event="meta-planning-bootstrap-pre-call-verify",
                                callable_present=hasattr(meta_planning, "start_self_improvement_cycle"),
                                planner_cls=str(planner_cls),
                                workflow_count=len(workflows),
                            ),
                        )
                        thread = meta_planning.start_self_improvement_cycle(
                            workflows,
                            event_bus=shared_event_bus,
                            interval=interval,
                            workflow_graph=workflow_graph_obj,
                        )
                        logger.info(
                            "🛰️ meta planning bootstrap call returned from start_self_improvement_cycle()",
                            extra=log_record(
                                event="meta-planning-bootstrap-returned",
                                thread_is_none=thread is None,
                                thread_type=type(thread).__name__ if thread is not None else None,
                                thread_dir=list(sorted(set(dir(thread)) if thread is not None else [])),
                                workflow_count=len(workflows),
                                interval_seconds=interval,
                            ),
                        )
                        if thread is None:
                            logger.error(
                                "❌ meta planning bootstrap returned None thread",
                                extra=log_record(
                                    event="meta-planning-thread-none",
                                    planner_cls=str(planner_cls),
                                    workflow_ids=list(workflows.keys()),
                                ),
                            )
                            raise RuntimeError("meta planning bootstrap returned None")
                        logger.info(
                            "🛰️ meta planning bootstrap returned valid thread object; proceeding to post-call checks",
                            extra=log_record(
                                event="meta-planning-bootstrap-post-call",
                                thread_name=getattr(thread, "name", "unknown"),
                                daemon=getattr(thread, "daemon", None),
                                alive=getattr(thread, "is_alive", lambda: False)(),
                                planner_cls=str(planner_cls),
                                workflow_ids=list(workflows.keys()),
                                event_bus_type=type(shared_event_bus).__name__
                                if shared_event_bus is not None
                                else None,
                            ),
                        )
                        logger.info(
                            "✅ meta planning bootstrap call returned",
                            extra=log_record(
                                event="meta-planning-bootstrap-return",
                                thread_repr=repr(thread),
                                thread_name=getattr(thread, "name", "unknown"),
                                thread_ident=getattr(thread, "ident", None),
                            ),
                        )
                        logger.info(
                            "✅ meta planning thread object created",
                            extra=log_record(
                                event="meta-planning-thread-created",
                                thread_name=getattr(thread, "name", "unknown"),
                                daemon=getattr(thread, "daemon", None),
                                planner_cls=str(planner_cls),
                                workflow_count=len(workflows),
                                target=getattr(thread, "_target", None),
                            ),
                        )
                        logger.info(
                            "✅ meta planning thread attributes captured",
                            extra=log_record(
                                event="meta-planning-thread-attrs",
                                thread_name=getattr(thread, "name", "unknown"),
                                daemon=getattr(thread, "daemon", None),
                                alive=getattr(thread, "is_alive", lambda: False)(),
                                thread_ident=getattr(thread, "ident", None),
                            ),
                        )
                        logger.info(
                            "✅ meta planning bootstrap pipeline completed; preparing start() call",
                            extra=log_record(
                                event="meta-planning-bootstrap-finished",
                                thread_name=getattr(thread, "name", "unknown"),
                                daemon=getattr(thread, "daemon", None),
                                planner_cls=str(planner_cls),
                                alive_pre_start=getattr(thread, "is_alive", lambda: False)(),
                            ),
                        )
                    except Exception as exc:
                        logger.exception(
                            "❌ meta planning loop bootstrap failed; thread object missing",
                            extra=log_record(
                                event="meta-loop-error",
                                workflow_count=len(workflows),
                                planner_cls=str(planner_cls),
                                interval_seconds=interval,
                                error_type=type(exc).__name__,
                                error_message=str(exc),
                            ),
                        )
                        logger.exception(
                            "❌ meta planning bootstrap returned invalid thread",
                            extra=log_record(
                                event="meta-planning-thread-invalid",
                                planner_cls=str(planner_cls),
                                workflow_count=len(workflows),
                                error_type=type(exc).__name__,
                            ),
                        )
                        logger.exception(
                            "failed to initialize meta planning loop; sandbox launch halted",
                            extra=log_record(event="meta-loop-error"),
                        )
                        sys.exit(1)

                    try:
                        logger.info(
                            "🧭 entering meta planning thread.start() block",  # explicit boundary marker
                            extra=log_record(
                                event="meta-planning-thread-start-boundary",
                                thread_name=getattr(thread, "name", "unknown"),
                                daemon=getattr(thread, "daemon", None),
                                alive_pre=getattr(thread, "is_alive", lambda: False)(),
                            ),
                        )
                        logger.info(
                            "🔧 attempting to start meta planning loop thread",
                            extra=log_record(
                                event="meta-planning-start-attempt",
                                thread_name=getattr(thread, "name", "unknown"),
                                daemon=getattr(thread, "daemon", None),
                                planner_cls=str(planner_cls),
                                workflow_count=len(workflows),
                            ),
                        )
                        thread.start()
                        logger.info(
                            "✅ thread.start() invoked successfully for meta planning loop",
                            extra=log_record(
                                event="meta-planning-thread-start-invoked",
                                thread_name=getattr(thread, "name", "unknown"),
                            ),
                        )
                        logger.info(
                            "✅ meta planning thread start invoked",
                            extra=log_record(
                                event="meta-planning-start-invoke",
                                thread_name=getattr(thread, "name", "unknown"),
                                daemon=getattr(thread, "daemon", None),
                            ),
                        )
                        logger.info(
                            "✅ meta planning loop thread started successfully",
                            extra=log_record(
                                event="meta-planning-start",
                                thread_name=getattr(thread, "name", "unknown"),
                                is_alive=getattr(thread, "is_alive", lambda: False)(),
                                planner_cls=str(planner_cls),
                                workflow_count=len(workflows),
                                workflow_graph_present=workflow_graph_obj is not None,
                            ),
                        )
                        logger.info(
                            "✅ meta planning thread alive status confirmed",
                            extra=log_record(
                                event="meta-planning-thread-alive",
                                thread_name=getattr(thread, "name", "unknown"),
                                is_alive=getattr(thread, "is_alive", lambda: False)(),
                            ),
                        )
                        alive_state = getattr(thread, "is_alive", lambda: False)()
                        if not alive_state:
                            logger.error(
                                "❌ meta planning loop thread reported not alive after start",
                                extra=log_record(
                                    event="meta-planning-thread-not-alive",
                                    thread_name=getattr(thread, "name", "unknown"),
                                    planner_cls=str(planner_cls),
                                    workflow_count=len(workflows),
                                ),
                            )
                            raise RuntimeError("meta planning loop thread failed to stay alive")
                        logger.info(
                            "✅ meta planning loop thread alive verification passed",
                            extra=log_record(
                                event="meta-planning-thread-alive-verified",
                                thread_name=getattr(thread, "name", "unknown"),
                                planner_cls=str(planner_cls),
                                workflow_count=len(workflows),
                            ),
                        )
                        logger.info(
                            "✅ meta planning loop start confirmed; orchestrator warm-up next",
                            extra=log_record(
                                event="meta-planning-start-confirm",
                                thread_name=getattr(thread, "name", "unknown"),
                                daemon=getattr(thread, "daemon", None),
                                is_alive=getattr(thread, "is_alive", lambda: False)(),
                            ),
                        )
                        logger.info(
                            "🎯 meta planning start block completed without exceptions; handing off to orchestrator",
                            extra=log_record(
                                event="meta-planning-start-block-complete",
                                thread_name=getattr(thread, "name", "unknown"),
                                alive=getattr(thread, "is_alive", lambda: False)(),
                                planner_cls=str(planner_cls),
                                workflow_ids=list(workflows.keys()),
                            ),
                        )
                    except Exception:
                        logger.exception(
                            "❌ meta planning loop thread failed to start",  # emoji for quick scanning
                            extra=log_record(event="meta-planning-start-error"),
                        )
                        sys.exit(1)
                else:
                    logger.error(
                        "❌ gating red: prelaunch ROI or planner checks failed; aborting meta-planning start",
                        extra=log_record(
                            event="meta-planning-gate-red-summary",
                            ready_to_launch=ready_to_launch,
                            roi_backoff=roi_backoff_triggered,
                            planner_available=planner_cls is not None,
                            workflows_present=bool(workflows),
                            readiness_error=readiness_error,
                        ),
                    )
                    logger.error(
                        "❌ meta planning loop launch gate failed",  # emoji for quick scanning
                        extra=log_record(
                            event="meta-planning-gate-failed",
                            roi_backoff=roi_backoff_triggered,
                            ready_to_launch=ready_to_launch,
                            planner_available=planner_cls is not None,
                            workflow_count=len(workflows),
                            readiness_error=readiness_error,
                        ),
                    )
                    logger.error(
                        "❌ readiness gate failed; meta planning launch conditions not met",
                        extra=log_record(
                            event="meta-planning-readiness-failure",
                            roi_backoff=roi_backoff_triggered,
                            ready_to_launch=ready_to_launch,
                            planner_available=planner_cls is not None,
                            workflow_count=len(workflows),
                            workflow_ids=list(workflows.keys()),
                        ),
                    )
                    failure_reason = readiness_error or (
                        "workflows did not reach ROI stagnation; sandbox launch aborted"
                    )
                    logger.error(
                        "❌ launch gate red; blocking meta planning loop",  # emoji for quick scanning
                        extra=log_record(
                            event="meta-planning-gate-red",
                            planner_available=planner_cls is not None,
                            workflow_count=len(workflows),
                            roi_backoff=roi_backoff_triggered,
                            ready_to_launch=ready_to_launch,
                            readiness_error=readiness_error,
                        ),
                    )
                    logger.error(
                        "❌ meta planning launch vetoed after readiness evaluation",
                        extra=log_record(
                            event="meta-planning-veto",
                            reason=failure_reason,
                            planner_available=planner_cls is not None,
                            workflow_count=len(workflows),
                            roi_backoff=roi_backoff_triggered,
                        ),
                    )
                    logger.error(
                        "❌ readiness gate blocked meta planning loop",
                        extra=log_record(
                            event="meta-planning-ready-false",
                            reason=failure_reason,
                            roi_backoff=roi_backoff_triggered,
                            ready_to_launch=ready_to_launch,
                        ),
                    )
                    logger.error(
                        "meta planning loop not started: %s",
                        failure_reason,
                        extra=log_record(
                            event="meta-planning-skipped",
                            reason=failure_reason,
                            workflow_count=len(workflows),
                            planner_available=planner_cls is not None,
                        ),
                    )
                    sys.exit(1)
                logger.info(
                    "preseeded bootstrap context in use; pipeline and manager are cached",
                    extra=log_record(event="bootstrap-preseed"),
                )

                if ready_to_launch:
                    try:
                        orchestrator = SandboxOrchestrator(
                            workflows,
                            logger=logger,
                            loop_interval=float(os.getenv("GLOBAL_ORCHESTRATOR_INTERVAL", "30")),
                            diminishing_threshold=float(
                                os.getenv("GLOBAL_ROI_DIMINISHING_THRESHOLD", "0.01")
                            ),
                            patience=int(os.getenv("GLOBAL_ROI_PATIENCE", "3")),
                        )
                        logger.info(
                            "✅ sandbox orchestrator object created",  # emoji for quick scanning
                            extra=log_record(
                                event="orchestrator-created",
                                workflow_count=len(workflows),
                                loop_interval=os.getenv("GLOBAL_ORCHESTRATOR_INTERVAL", "30"),
                                diminishing_threshold=os.getenv("GLOBAL_ROI_DIMINISHING_THRESHOLD", "0.01"),
                                patience=os.getenv("GLOBAL_ROI_PATIENCE", "3"),
                            ),
                        )
                    except Exception:
                        logger.exception(
                            "❌ failed to build sandbox orchestrator",  # emoji for quick scanning
                            extra=log_record(event="orchestrator-build-error"),
                        )
                        sys.exit(1)

                    try:
                        orchestrator_thread = threading.Thread(
                            target=orchestrator.run,
                            name="sandbox-orchestrator",
                            daemon=True,
                        )
                        orchestrator_thread.start()
                        logger.info(
                            "✅ sandbox orchestrator started",  # emoji for quick scanning
                            extra=log_record(
                                event="orchestrator-start",
                                thread_name="sandbox-orchestrator",
                                is_alive=orchestrator_thread.is_alive(),
                                workflow_count=len(workflows),
                            ),
                        )
                    except Exception:
                        logger.exception(
                            "❌ failed to start sandbox orchestrator thread",  # emoji for quick scanning
                            extra=log_record(event="orchestrator-start-error"),
                        )
                        sys.exit(1)
            except Exception:  # pragma: no cover - defensive bootstrap hint
                logger.exception(
                    "failed to preseed bootstrap context before bot loading",
                    extra=log_record(event="bootstrap-preseed-error"),
                )

        if args.health_check:
            bootstrap_environment(
                initialize=False,
                enforce_dependencies=False,
            )
            try:
                health_snapshot = sandbox_health()
            except Exception as exc:  # pragma: no cover - defensive fallback
                sandbox_crashes_total.inc()
                sandbox_last_failure_ts.set(time.time())
                logger.exception(
                    "Sandbox health probe failed", extra=log_record(event="health-error")
                )
                failures = [f"health probe failed: {exc}"]
                _emit_health_report(
                    {"error": str(exc)},
                    healthy=False,
                    failures=failures,
                )
                shutdown_autonomous_sandbox()
                logger.info("sandbox shutdown", extra=log_record(event="shutdown"))
                sys.exit(2)

            logger.info(
                "Sandbox health", extra=log_record(health=health_snapshot)
            )
            healthy, failures = _evaluate_health(
                health_snapshot,
                dependency_mode=_resolve_dependency_mode(settings),
            )
            _emit_health_report(
                health_snapshot,
                healthy=healthy,
                failures=failures,
            )
            shutdown_autonomous_sandbox()
            logger.info("sandbox shutdown", extra=log_record(event="shutdown"))
            if not healthy:
                logger.error(
                    "Sandbox health check failed: %s",
                    "; ".join(failures) if failures else "unknown reason",
                )
                sys.exit(2)
            return
        if roi_backoff_triggered:
            logger.warning(
                "sandbox launch halted due to ROI backoff",
                extra=log_record(event="roi-backoff-block", correlation_id=cid),
            )
            return
        if not ready_to_launch:
            logger.warning(
                "sandbox launch gated until workflows reach diminishing returns",
                extra=log_record(event="diminishing-returns-pending"),
            )
            return
        print("[start_autonomous_sandbox] launching sandbox", flush=True)
        launch_sandbox()
        print("[start_autonomous_sandbox] sandbox exited", flush=True)
        logger.info("sandbox shutdown", extra=log_record(event="shutdown"))
    except Exception:  # pragma: no cover - defensive catch
        sandbox_crashes_total.inc()
        sandbox_last_failure_ts.set(time.time())
        logger.exception("Failed to launch sandbox", extra=log_record(event="failure"))
        sys.exit(1)
    finally:
        try:
            shutdown_autonomous_sandbox()
        except Exception:
            logger.exception(
                "sandbox shutdown failed", extra=log_record(event="shutdown-error")
            )
        set_correlation_id(None)


if __name__ == "__main__":
    main()
