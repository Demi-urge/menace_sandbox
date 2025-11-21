"""Entry point for launching the autonomous sandbox.

This wrapper bootstraps the environment and model paths automatically before
starting the sandbox. It captures startup exceptions and allows the log level
to be configured via ``SandboxSettings`` or overridden on the command line
without requiring any manual post-launch edits.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

def _maybe_exit_early_for_help(argv: Sequence[str]) -> None:
    """Print CLI help without triggering heavyweight imports.

    The sandbox bootstrapping stack performs extensive dynamic imports that can
    span several seconds even when callers only request ``--help``. To keep the
    help path responsive and side-effect free, we short-circuit before loading
    the rest of the module graph whenever a help flag is present.
    """

    if not any(arg in {"-h", "--help"} for arg in argv):
        return

    parser = argparse.ArgumentParser(description="Launch the autonomous sandbox")
    parser.add_argument("--log-level", dest="log_level", default="INFO")
    parser.add_argument("--health-check", action="store_true")
    parser.add_argument("--monitor-roi-backoff", action="store_true")
    parser.add_argument("--bootstrap-timeout", type=float, default=300.0)
    parser.add_argument(
        "--include-orphans",
        dest="include_orphans",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--recursive-orphan-scan",
        dest="recursive_orphan_scan",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.print_help()
    sys.exit(0)


_maybe_exit_early_for_help(sys.argv[1:])

if "--health-check" in sys.argv[1:]:
    if not os.getenv("SANDBOX_DEPENDENCY_MODE"):
        os.environ["SANDBOX_DEPENDENCY_MODE"] = "minimal"
    # Disable long-running monitoring loops during the lightweight health
    # probe so the command terminates promptly even when background services
    # would normally bootstrap DataBot.
    os.environ.setdefault("MENACE_SANDBOX_MODE", "health_check")
    os.environ.setdefault("MENACE_DISABLE_MONITORING", "1")


def _ensure_package_namespace() -> None:
    """Expose this repository as the ``menace_sandbox`` package when uninstalled.

    The sandbox entrypoints rely on absolute imports such as
    ``menace_sandbox.bot_registry``. When executed from a checked-out
    repository without installation, those imports fail because Python does
    not automatically treat the repo root as a package. This helper builds and
    registers a module object backed by ``__init__.py`` so downstream imports
    behave identically to an installed distribution while keeping dynamic
    path resolution intact.
    """

    package_name = "menace_sandbox"
    package_root = Path(__file__).resolve().parent
    init_path = package_root / "__init__.py"

    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

    existing = sys.modules.get(package_name)
    if existing is None or getattr(existing, "__file__", None) != str(init_path):
        spec = importlib.util.spec_from_file_location(package_name, init_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            module.__path__ = [str(package_root)]
            sys.modules[package_name] = module
            spec.loader.exec_module(module)
    else:
        module_path = getattr(existing, "__path__", None)
        if isinstance(module_path, list) and str(package_root) not in module_path:
            module_path.append(str(package_root))


_ensure_package_namespace()

from logging_utils import get_logger, setup_logging, set_correlation_id, log_record
from sandbox_settings import SandboxSettings
from dependency_health import DependencyMode, resolve_dependency_mode
from sandbox.preseed_bootstrap import BOOTSTRAP_PROGRESS, initialize_bootstrap_context
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


def _emit_meta_trace(logger: logging.Logger, message: str, **details: Any) -> None:
    """Log and print a dense meta-planning breadcrumb for immediate visibility."""

    payload = log_record(event="meta-trace", **details)
    logger.info(message, extra=payload)
    summary_bits = ", ".join(f"{k}={v}" for k, v in sorted(details.items()))
    print(f"[META-TRACE] {message} :: {summary_bits}", flush=True)


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

    if not history_path.exists():
        LOGGER.warning(
            "prelaunch ROI baseline unavailable: history file missing; forcing bootstrap mode",
            extra=log_record(
                event="roi-baseline-missing",
                history_path=str(history_path),
            ),
        )
        return False

    try:
        raw = history_path.read_text()
    except Exception:
        LOGGER.exception(
            "prelaunch ROI baseline unavailable: unable to read history file; forcing bootstrap mode",
            extra=log_record(
                event="roi-baseline-read-error",
                history_path=str(history_path),
            ),
        )
        return False

    if not raw.strip():
        LOGGER.warning(
            "prelaunch ROI baseline unavailable: history file is empty; forcing bootstrap mode",
            extra=log_record(
                event="roi-baseline-empty",
                history_path=str(history_path),
            ),
        )
        return False

    try:
        data = json.loads(raw)
    except Exception:
        LOGGER.warning(
            "prelaunch ROI baseline unavailable: history file contains invalid JSON; forcing bootstrap mode",
            extra=log_record(
                event="roi-baseline-invalid-json",
                history_path=str(history_path),
            ),
        )
        return False

    if not isinstance(data, Mapping):
        LOGGER.warning(
            "prelaunch ROI baseline unavailable: unexpected history format; forcing bootstrap mode",
            extra=log_record(
                event="roi-baseline-invalid-format",
                history_path=str(history_path),
                data_type=type(data).__name__,
            ),
        )
        return False

    valid_entries = 0
    for _, values in data.items():
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            continue

        for value in values:
            try:
                float(value)
            except Exception:
                continue
            valid_entries += 1
            break

    if not valid_entries:
        LOGGER.warning(
            "prelaunch ROI baseline unavailable: no valid ROI entries found; forcing bootstrap mode",
            extra=log_record(
                event="roi-baseline-empty-data",
                history_path=str(history_path),
            ),
        )
        return False

    return True


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
    print(
        "[META-TRACE] initializing meta planner class=%s continuous_monitor=%s cycle_budget=%s"
        % (
            getattr(planner_cls, "__name__", str(planner_cls)),
            continuous_monitor,
            cycle_budget,
        ),
        flush=True,
    )
    try:
        planner = planner_cls(context_builder=create_context_builder())
        logger.info(
            "✅ meta planner initialized for ROI coordination",
            extra=log_record(event="meta-coordinator-init-success"),
        )
        _emit_meta_trace(
            logger,
            "meta planner instantiated",
            planner_class=getattr(planner_cls, "__name__", str(planner_cls)),
            continuous_monitor=continuous_monitor,
            cycle_budget=cycle_budget,
            workflow_count=len(workflows),
        )
        print(
            "[META-TRACE] meta planner instantiated with workflows=%d context_builder_ready=%s"
            % (len(workflows), hasattr(planner, "context_builder")),
            flush=True,
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
            print(
                "[META-TRACE] planner attribute %s set to %s" % (name, value),
                flush=True,
            )
            _emit_meta_trace(
                logger,
                "planner attribute updated",
                setting=name,
                value=value,
                planner_class=getattr(planner_cls, "__name__", str(planner_cls)),
            )
        else:
            logger.debug(
                "ℹ️ planner setting skipped; attribute missing",
                extra=log_record(event="meta-coordinator-setting-skipped", setting=name),
            )
            _emit_meta_trace(
                logger,
                "planner attribute missing; skipped update",
                setting=name,
                planner_class=getattr(planner_cls, "__name__", str(planner_cls)),
            )
            print(
                "[META-TRACE] planner missing attribute %s; leaving default" % name,
                flush=True,
            )

    diminishing: set[str] = set()
    roi_backoff_triggered = False
    budget = cycle_budget or max(len(workflows) * streak_required * 2, 3)
    print(
        "[META-TRACE] planner cycle budget established at %d (workflows=%d streak_required=%d threshold=%.4f)"
        % (budget, len(workflows), streak_required, threshold),
        flush=True,
    )

    for cycle in range(budget):
        _emit_meta_trace(
            logger,
            "meta planner coordination cycle start",
            cycle=cycle,
            budget=budget,
            workflow_count=len(workflows),
            diminishing=len(diminishing),
        )
        try:
            records = planner.discover_and_persist(workflows)
            logger.info(
                "meta planner cycle executed",  # dense trace per cycle
                extra=log_record(
                    event="meta-coordinator-cycle",
                    cycle=cycle,
                    budget=budget,
                    record_count=len(records) if records else 0,
                    diminishing=len(diminishing),
                    workflows=list(workflows.keys()),
                ),
            )
            print(
                "[META-TRACE] planner cycle %d completed; records=%d diminishing=%d"
                % (cycle, len(records) if records else 0, len(diminishing)),
                flush=True,
            )
            print(
                "[META-TRACE] planner cycle %d outputs=%s" % (cycle, records),
                flush=True,
            )
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
            _emit_meta_trace(
                logger,
                "meta planner returned no records",
                cycle=cycle,
                diminishing=len(diminishing),
                workflow_count=len(workflows),
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
            _emit_meta_trace(
                logger,
                "meta planning record processed",
                workflow_id=chain_id,
                roi_gain=roi_gain,
                roi_delta=roi_delta,
                stagnated=stagnated,
                streak=streak,
                cycle=cycle,
                diminishing=len(diminishing),
            )
            print(
                "[META-TRACE] record processed; chain=%s roi_gain=%.4f roi_delta=%.4f stagnated=%s streak=%d"
                % (chain_id, roi_gain, roi_delta, stagnated, streak),
                flush=True,
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
                _emit_meta_trace(
                    logger,
                    "roi backoff triggered",
                    workflow_id=chain_id,
                    roi_delta=roi_delta,
                    threshold=threshold,
                    streak=streak,
                    cycle=cycle,
                )
                break

        if roi_backoff_triggered or len(diminishing) >= len(workflows):
            print(
                "[META-TRACE] planner terminating early; roi_backoff=%s diminishing=%d/%d"
                % (roi_backoff_triggered, len(diminishing), len(workflows)),
                flush=True,
            )
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
        print(
            "[META-TRACE] diminishing returns incomplete; achieved=%d total=%d"
            % (len(diminishing), len(workflows)),
            flush=True,
        )
    else:
        print(
            "[META-TRACE] diminishing returns reached for all workflows; ready=%s backoff=%s"
            % (ready, roi_backoff_triggered),
            flush=True,
        )

    _emit_meta_trace(
        logger,
        "meta coordination completed",
        ready=ready,
        roi_backoff_triggered=roi_backoff_triggered,
        diminishing=len(diminishing),
        workflow_count=len(workflows),
        cycles_budget=budget,
    )
    print(
        "[META-TRACE] meta coordination complete; ready=%s roi_backoff=%s diminishing=%d/%d"
        % (ready, roi_backoff_triggered, len(diminishing), len(workflows)),
        flush=True,
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

    bootstrap_timeout_default = 300.0
    env_bootstrap_timeout = os.getenv("BOOTSTRAP_CONTEXT_TIMEOUT")
    if env_bootstrap_timeout:
        try:
            bootstrap_timeout_default = float(env_bootstrap_timeout)
        except ValueError:
            print(
                "[WARN] BOOTSTRAP_CONTEXT_TIMEOUT is not a number; using 300s default",
                flush=True,
            )

    parser = argparse.ArgumentParser(description="Launch the autonomous sandbox")
    parser.add_argument(
        "--log-level",
        dest="log_level",
        default=None,
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
        "--bootstrap-timeout",
        type=float,
        default=bootstrap_timeout_default,
        help="Maximum seconds to wait for initialize_bootstrap_context before failing",
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

    if args.health_check and not os.getenv("SANDBOX_DEPENDENCY_MODE"):
        os.environ["SANDBOX_DEPENDENCY_MODE"] = "minimal"

    settings = SandboxSettings()
    # Automatically configure the environment before proceeding so the caller
    # does not need to pre-populate configuration files or model paths.
    auto_configure_env(settings)
    # Reload settings to pick up any values written by ``auto_configure_env``.
    settings = SandboxSettings()

    print(
        f"[DEBUG] Parsed args: {args}; health_check={getattr(args, 'health_check', None)}",
        flush=True,
    )

    resolved_log_level = args.log_level or settings.sandbox_log_level
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
        setup_logging(level=resolved_log_level)
    else:
        root_logger.setLevel(
            getattr(logging, str(resolved_log_level).upper(), logging.INFO)
        )
    cid = f"sas-{uuid.uuid4()}"
    set_correlation_id(cid)
    logger = get_logger(__name__)
    sandbox_restart_total.labels(service="start_autonomous", reason="launch").inc()
    logger.info("sandbox start", extra=log_record(event="start"))

    ready_to_launch = True
    roi_backoff_triggered = False
    failure_reasons: list[str] = []

    bootstrap_stop_event: threading.Event | None = None
    bootstrap_thread: threading.Thread | None = None

    try:
        if not args.health_check:
            last_pre_meta_trace_step = "entering non-health-check bootstrap block"
            try:
                try:
                    last_pre_meta_trace_step = "initialize_bootstrap_context entry"
                    print(
                        "[DEBUG] About to call initialize_bootstrap_context()",
                        flush=True,
                    )
                    logger.info(
                        "initialize_bootstrap_context starting",
                        extra=log_record(
                            event="bootstrap-context-start",
                            health_check=args.health_check,
                        ),
                    )
                    try:
                        last_pre_meta_trace_step = "initialize_bootstrap_context invocation"
                        bootstrap_context_result: dict[str, Any] | None = None
                        bootstrap_error: BaseException | None = None
                        bootstrap_start = time.monotonic()
                        bootstrap_deadline = bootstrap_start + args.bootstrap_timeout
                        bootstrap_stop_event = threading.Event()
                        critical_grace_extension_applied = False
                        critical_grace_extension_seconds = 15.0

                        def _run_bootstrap() -> None:
                            nonlocal bootstrap_context_result, bootstrap_error
                            try:
                                bootstrap_context_result = initialize_bootstrap_context(
                                    stop_event=bootstrap_stop_event,
                                    bootstrap_deadline=bootstrap_deadline,
                                )
                            except BaseException as exc:  # pragma: no cover - propagate errors
                                bootstrap_error = exc

                        bootstrap_thread = threading.Thread(
                            target=_run_bootstrap,
                            name="bootstrap-context",
                            daemon=True,
                        )
                        bootstrap_thread.start()
                        initial_wait = max(args.bootstrap_timeout - 5.0, 0.0)
                        bootstrap_thread.join(initial_wait)

                        bootstrap_completed = False

                        last_bootstrap_step = BOOTSTRAP_PROGRESS.get(
                            "last_step", "unknown"
                        )
                        if last_bootstrap_step == "bootstrap_complete":
                            bootstrap_thread.join()
                            bootstrap_completed = True
                            _emit_meta_trace(
                                logger,
                                "bootstrap thread finished; fast-forwarding to meta planning",
                                last_step=last_bootstrap_step,
                                elapsed=round(time.monotonic() - bootstrap_start, 3),
                            )
                        elif bootstrap_thread.is_alive():
                            critical_bootstrap_steps = {
                                "self_coding_engine",
                                "prepare_pipeline",
                                "internalize_coding_bot",
                                "promote_pipeline",
                                "seed_final_context",
                                "push_final_context",
                            }
                            finalization_steps = {
                                "prepare_pipeline",
                                "internalize_coding_bot",
                                "promote_pipeline",
                                "seed_final_context",
                                "push_final_context",
                            }
                            finalization_grace_seconds = 5.0
                            finalization_grace_applied = False
                            while bootstrap_thread.is_alive():
                                time_remaining = bootstrap_deadline - time.monotonic()
                                last_bootstrap_step = BOOTSTRAP_PROGRESS.get(
                                    "last_step", "unknown"
                                )
                                if (
                                    time_remaining <= 3.0
                                    and last_bootstrap_step in finalization_steps
                                    and not finalization_grace_applied
                                ):
                                    LOGGER.warning(
                                        "bootstrap in %s with %.1fs remaining; applying "
                                        "finalization grace window",
                                        last_bootstrap_step,
                                        max(time_remaining, 0.0),
                                        extra=log_record(
                                            event="bootstrap-finalization-grace",
                                            last_bootstrap_step=last_bootstrap_step,
                                            time_remaining=max(time_remaining, 0.0),
                                            grace_seconds=finalization_grace_seconds,
                                        ),
                                    )
                                    finalization_grace_applied = True
                                    bootstrap_deadline += finalization_grace_seconds
                                    time_remaining = bootstrap_deadline - time.monotonic()
                                if (
                                    time_remaining <= 5.0
                                    and last_bootstrap_step == "embedder_preload"
                                ):
                                    LOGGER.warning(
                                        "bootstrap embedder still active with %.1fs remaining; "
                                        "requesting early self-coding abort",
                                        max(time_remaining, 0.0),
                                    )
                                    bootstrap_stop_event.set()

                                if last_bootstrap_step in critical_bootstrap_steps:
                                    if (
                                        time_remaining <= 10.0
                                        and not critical_grace_extension_applied
                                    ):
                                        LOGGER.warning(
                                            "bootstrap still in %s with %.1fs remaining; "
                                            "granting one-time grace window",
                                            last_bootstrap_step,
                                            max(time_remaining, 0.0),
                                            extra=log_record(
                                                event="bootstrap-deadline-guard",
                                                last_bootstrap_step=last_bootstrap_step,
                                                time_remaining=max(time_remaining, 0.0),
                                                grace_applied=True,
                                                grace_seconds=critical_grace_extension_seconds,
                                            ),
                                        )
                                        critical_grace_extension_applied = True
                                        bootstrap_deadline += critical_grace_extension_seconds
                                        time_remaining = bootstrap_deadline - time.monotonic()
                                    elif time_remaining <= 2.0 and not (
                                        last_bootstrap_step in finalization_steps
                                        and finalization_grace_applied
                                    ):
                                        LOGGER.error(
                                            "bootstrap still in %s with %.1fs remaining after grace; "
                                            "signaling stop event",
                                            last_bootstrap_step,
                                            max(time_remaining, 0.0),
                                            extra=log_record(
                                                event="bootstrap-deadline-guard",
                                                last_bootstrap_step=last_bootstrap_step,
                                                time_remaining=max(time_remaining, 0.0),
                                                grace_applied=critical_grace_extension_applied,
                                            ),
                                        )
                                        bootstrap_stop_event.set()

                                wait_time = min(max(time_remaining, 0.0), 1.0)
                                if wait_time <= 0:
                                    break
                                bootstrap_thread.join(wait_time)

                        if not bootstrap_completed and not bootstrap_thread.is_alive():
                            last_bootstrap_step = BOOTSTRAP_PROGRESS.get(
                                "last_step", "unknown"
                            )
                            if last_bootstrap_step == "bootstrap_complete":
                                bootstrap_completed = True
                                _emit_meta_trace(
                                    logger,
                                    "bootstrap completed after guarded wait; entering meta planning",
                                    last_step=last_bootstrap_step,
                                    elapsed=round(time.monotonic() - bootstrap_start, 3),
                                )
                            else:
                                LOGGER.error(
                                    "bootstrap thread exited without reporting completion (last_step=%s)",
                                    last_bootstrap_step,
                                    extra=log_record(
                                        event="bootstrap-missing-meta-trace",
                                        last_bootstrap_step=last_bootstrap_step,
                                        elapsed=round(time.monotonic() - bootstrap_start, 3),
                                    ),
                                )
                                raise RuntimeError(
                                    "bootstrap thread finished without reaching bootstrap_complete; "
                                    f"last_step={last_bootstrap_step}"
                                )

                        elapsed = time.monotonic() - bootstrap_start
                        if bootstrap_thread.is_alive():
                            last_bootstrap_step = BOOTSTRAP_PROGRESS.get(
                                "last_step", "unknown"
                            )
                            print(
                                "[BOOTSTRAP-TRACE] initialize_bootstrap_context exceeded "
                                f"timeout after {elapsed:.1f}s (limit={args.bootstrap_timeout}s, "
                                f"last_step={last_bootstrap_step})",
                                flush=True,
                            )
                            bootstrap_stop_event.set()
                            logger.error(
                                "initialize_bootstrap_context exceeded timeout",
                                extra=log_record(
                                    event="bootstrap-context-timeout",
                                    elapsed=elapsed,
                                    timeout=args.bootstrap_timeout,
                                    last_bootstrap_step=last_bootstrap_step,
                                    last_pre_meta_trace_step=last_pre_meta_trace_step,
                                ),
                            )
                            try:
                                shutdown_autonomous_sandbox(timeout=5)
                            except Exception:  # pragma: no cover - best effort cleanup
                                logger.exception(
                                    "cleanup after bootstrap timeout failed",
                                    extra=log_record(event="bootstrap-timeout-cleanup-error"),
                                )
                            try:
                                bootstrap_thread.join(2)
                            except Exception:
                                logger.debug(
                                    "bootstrap thread join after timeout raised",
                                    exc_info=True,
                                )
                            raise TimeoutError(
                                "initialize_bootstrap_context exceeded timeout; "
                                f"last_step={last_bootstrap_step} elapsed={elapsed:.1f}s"
                            )

                        if bootstrap_error:
                            raise bootstrap_error

                        bootstrap_context = bootstrap_context_result
                    except Exception as bootstrap_exc:
                        last_bootstrap_step = BOOTSTRAP_PROGRESS.get(
                            "last_step", "unknown"
                        )
                        print(
                            "[DEBUG] initialize_bootstrap_context raised: "
                            f"{bootstrap_exc} (last_step={last_bootstrap_step})",
                            flush=True,
                        )
                        print(
                            "[BOOTSTRAP-TRACE] bootstrap failed before meta-trace; "
                            f"last_step={last_bootstrap_step}",
                            flush=True,
                        )
                        logger.exception(
                            "initialize_bootstrap_context encountered an exception",
                            extra=log_record(
                                event="bootstrap-context-error",
                                last_bootstrap_step=last_bootstrap_step,
                                last_pre_meta_trace_step=last_pre_meta_trace_step,
                            ),
                        )
                        raise
                    print(
                        "[DEBUG] initialize_bootstrap_context completed successfully",
                        flush=True,
                    )
                    logger.info(
                        "initialize_bootstrap_context completed",
                        extra=log_record(event="bootstrap-context-complete"),
                    )
                    os.environ.setdefault("META_PLANNING_LOOP", "1")
                    os.environ.setdefault("META_PLANNING_INTERVAL", "10")
                    os.environ.setdefault("META_IMPROVEMENT_THRESHOLD", "0.01")
                    _emit_meta_trace(
                        logger,
                        "preparing meta planning environment",
                        loop=os.environ.get("META_PLANNING_LOOP"),
                        interval=os.environ.get("META_PLANNING_INTERVAL"),
                        improvement_threshold=os.environ.get("META_IMPROVEMENT_THRESHOLD"),
                    )
                    last_pre_meta_trace_step = "importing self_improvement.meta_planning"
                    from self_improvement import meta_planning
                    print(
                        "[META-TRACE] meta_planning module import completed; capturing module attributes",
                        flush=True,
                    )
                    logger.info(
                        "meta_planning module import completed; enumerating attributes",
                        extra=log_record(
                            event="meta-planning-import-finished",
                            attr_count=len(dir(meta_planning)),
                            attrs_preview=list(sorted(dir(meta_planning)))[:25],
                        ),
                    )
                    logger.info(
                        "meta_planning module imported for autonomous sandbox",
                        extra=log_record(
                            event="meta-planning-import",
                            module=str(meta_planning),
                            module_dir=list(sorted(dir(meta_planning))),
                        ),
                    )
                    _emit_meta_trace(
                        logger,
                        "meta planning module imported",
                        module=str(meta_planning),
                        meta_planning_interval=os.environ.get("META_PLANNING_INTERVAL"),
                    )
                    last_pre_meta_trace_step = "importing self_improvement_cycle"
                    from self_improvement.meta_planning import (  # noqa: F401
                        self_improvement_cycle,
                    )
                    print(
                        "[META-TRACE] self_improvement_cycle imported; meta planner wiring begins",
                        flush=True,
                    )
                    logger.info(
                        "self_improvement_cycle imported; preparing to reload settings",
                        extra=log_record(
                            event="meta-planning-cycle-imported",
                            module_has_cycle=hasattr(meta_planning, "self_improvement_cycle"),
                        ),
                    )
    
                    last_pre_meta_trace_step = "reloading meta_planning settings"
                    meta_planning.reload_settings(settings)
                    print(
                        "[META-TRACE] meta_planning.reload_settings invoked; settings synchronized",
                        flush=True,
                    )
                    logger.info(
                        "meta planning settings synchronized",
                        extra=log_record(
                            event="meta-planning-settings-reloaded",
                            include_orphans=settings.include_orphans,
                            recursive_orphans=settings.recursive_orphan_scan,
                            sandbox_log_level=settings.sandbox_log_level,
                        ),
                    )
                    _emit_meta_trace(
                        logger,
                        "meta planning settings reloaded",
                        include_orphans=settings.include_orphans,
                        recursive_orphans=settings.recursive_orphan_scan,
                        log_level=settings.sandbox_log_level,
                    )
                    workflow_evolver = WorkflowEvolutionManager()
                    print(
                        "[META-TRACE] WorkflowEvolutionManager instantiated; preparing planner resolution",
                        flush=True,
                    )
                    _emit_meta_trace(
                        logger,
                        "workflow evolver instantiated for meta planning",
                        evolver_class=WorkflowEvolutionManager.__name__,
                    )
                    print(
                        "[META-TRACE] workflow evolver ready; resolving planner with force reload",
                        flush=True,
                    )
                    planner_cls = meta_planning.resolve_meta_workflow_planner(
                        force_reload=True
                    )
                    logger.info(
                        "meta workflow planner resolved",  # dense log for planner resolution
                        extra=log_record(
                            event="meta-planning-planner-resolved",
                            planner_cls=getattr(planner_cls, "__name__", str(planner_cls)),
                            force_reload=True,
                        ),
                    )
                    print(
                        "[META-TRACE] meta workflow planner resolution finished; class=%s"
                        % getattr(planner_cls, "__name__", str(planner_cls)),
                        flush=True,
                    )
                    logger.info(
                        "meta workflow planner resolution detailed trace",
                        extra=log_record(
                            event="meta-planning-planner-resolution-detail",
                            planner_cls=getattr(planner_cls, "__name__", str(planner_cls)),
                            planner_module=getattr(planner_cls, "__module__", None),
                            planner_dict=sorted(list(getattr(planner_cls, "__dict__", {}).keys())),
                        ),
                    )
                    _emit_meta_trace(
                        logger,
                        "meta workflow planner resolution attempted",
                        planner_resolved=planner_cls is not None,
                        planner_cls=getattr(planner_cls, "__name__", str(planner_cls)),
                    )
                    if planner_cls is None:
                        logger.error(
                            "MetaWorkflowPlanner not found; aborting sandbox launch",
                            extra=log_record(event="meta-planning-missing"),
                        )
                        print(
                            "[META-TRACE] planner resolution failed; aborting launch pipeline",
                            flush=True,
                        )
                        sys.exit(1)
    
                    interval = float(
                        os.getenv(
                            "META_PLANNING_INTERVAL",
                            getattr(settings, "meta_planning_interval", 10),
                        )
                    )
                    logger.info(
                        "meta planning cadence calculated",
                        extra=log_record(
                            event="meta-planning-interval",
                            interval=interval,
                            source_env=os.getenv("META_PLANNING_INTERVAL"),
                            settings_interval=getattr(settings, "meta_planning_interval", None),
                            settings_namespace=vars(settings),
                        ),
                    )
                    print(
                        "[META-TRACE] meta planning interval established at %.2fs" % interval,
                        flush=True,
                    )
                    logger.info(
                        "meta planning cadence fully resolved with environment and settings context",
                        extra=log_record(
                            event="meta-planning-interval-detail",
                            interval=interval,
                            env_interval=os.getenv("META_PLANNING_INTERVAL"),
                            env_loop=os.getenv("META_PLANNING_LOOP"),
                            improvement_threshold=os.getenv("META_IMPROVEMENT_THRESHOLD"),
                        ),
                    )
                    discovered_specs = []
                    try:
                        discovered_specs = discover_workflow_specs(logger=logger)
                        logger.info(
                            "workflow discovery completed",
                            extra=log_record(
                                event="workflow-discovery-complete",
                                discovered_count=len(discovered_specs),
                            ),
                        )
                        _emit_meta_trace(
                            logger,
                            "workflow discovery completed",
                            discovered_count=len(discovered_specs),
                            planner_cls=getattr(planner_cls, "__name__", str(planner_cls)),
                        )
                        print(
                            "[META-TRACE] workflow discovery finished; discovered=%d"
                            % len(discovered_specs),
                            flush=True,
                        )
                    except Exception:
                        logger.exception(
                            "failed to auto-discover workflow specs",
                            extra=log_record(event="workflow-discovery-error"),
                        )
                    print(
                        "[META-TRACE] workflow discovery post-processing; specs=%s"
                        % [spec.get("workflow_id") for spec in discovered_specs],
                        flush=True,
                    )
                    logger.info(
                        "workflow discovery snapshot",
                        extra=log_record(
                            event="workflow-discovery-snapshot",
                            discovered_ids=[spec.get("workflow_id") for spec in discovered_specs],
                            discovered_preview=discovered_specs[:3],
                        ),
                    )
    
                    orphan_specs: list[Mapping[str, Any]] = []
                    include_orphans = bool(
                        getattr(settings, "include_orphans", False)
                        and not getattr(settings, "disable_orphans", False)
                    )
                    recursive_orphans = bool(
                        getattr(settings, "recursive_orphan_scan", False)
                    )
                    logger.info(
                        "orphan inclusion parameters evaluated",
                        extra=log_record(
                            event="orphan-parameters",
                            include_orphans=include_orphans,
                            recursive_orphans=recursive_orphans,
                        ),
                    )
                    print(
                        "[META-TRACE] orphan settings finalized; include=%s recursive=%s"
                        % (include_orphans, recursive_orphans),
                        flush=True,
                    )
                    logger.info(
                        "orphan settings finalized for meta planning",
                        extra=log_record(
                            event="orphan-settings-finalized",
                            include_orphans=include_orphans,
                            recursive_orphans=recursive_orphans,
                            planner_cls=getattr(planner_cls, "__name__", str(planner_cls)),
                        ),
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
                            logger.info(
                                "orphan integration completed",
                                extra=log_record(
                                    event="orphan-integration-complete",
                                    orphan_modules=orphan_modules,
                                    orphan_spec_count=len(orphan_specs),
                                    recursive=recursive_orphans,
                                ),
                            )
                            print(
                                "[META-TRACE] orphan integration complete; modules=%s specs=%d"
                                % (orphan_modules, len(orphan_specs)),
                                flush=True,
                            )
                        except Exception:
                            logger.exception(
                                "startup orphan integration failed",
                                extra=log_record(event="startup-orphan-discovery"),
                            )
    
                    if include_orphans:
                        print(
                            "[META-TRACE] orphan discovery sequence completed; total specs now=%d" % len(orphan_specs),
                            flush=True,
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
                                    logger.info(
                                        "recursive orphan scan integrated modules",
                                        extra=log_record(
                                            event="recursive-orphan-scan",
                                            integrated=integrated,
                                            orphan_spec_count=len(orphan_specs),
                                        ),
                                    )
                                    print(
                                        "[META-TRACE] recursive orphan scan added modules=%s"
                                        % integrated,
                                        flush=True,
                                    )
                            except Exception:
                                logger.exception(
                                    "startup recursive orphan scan failed",
                                    extra=log_record(event="startup-orphan-recursive"),
                                )
    
                    _emit_meta_trace(
                        logger,
                        "orphan integration complete",
                        include_orphans=include_orphans,
                        recursive_orphans=recursive_orphans,
                        orphan_specs=len(orphan_specs),
                    )
                    print(
                        "[META-TRACE] orphan integration trace emitted; combined specs=%d"
                        % (len(discovered_specs) + len(orphan_specs)),
                        flush=True,
                    )
    
                    workflows, workflow_graph_obj = _build_self_improvement_workflows(
                        bootstrap_context,
                        settings,
                        workflow_evolver,
                        logger=logger,
                        discovered_specs=[*discovered_specs, *orphan_specs],
                    )
                    print(
                        "[META-TRACE] self-improvement workflows constructed; workflow_count=%d graph_nodes=%d"
                        % (
                            len(workflows),
                            len(getattr(workflow_graph_obj, "graph", {}) or {}),
                        ),
                        flush=True,
                    )
                    logger.info(
                        "self-improvement workflows constructed for meta planner",
                        extra=log_record(
                            event="workflows-constructed",
                            workflow_ids=list(workflows.keys()),
                            graph_summary=getattr(workflow_graph_obj, "graph", {}),
                        ),
                    )
                    _emit_meta_trace(
                        logger,
                        "workflows built for meta planning",
                        workflow_count=len(workflows),
                        graph_nodes=len(getattr(workflow_graph_obj, "graph", {})),
                        planner_cls=getattr(planner_cls, "__name__", str(planner_cls)),
                    )
                    print(
                        "[META-TRACE] workflow build meta trace emitted; planner=%s"
                        % getattr(planner_cls, "__name__", str(planner_cls)),
                        flush=True,
                    )
                    logger.info(
                        "workflow registration result",
                        extra=log_record(
                            event="workflow-registration",
                            workflow_count=len(workflows),
                            planner_available=planner_cls is not None,
                        ),
                    )
                    logger.info(
                        "workflow registration snapshot",
                        extra=log_record(
                            event="workflow-registration-detail",
                            workflow_keys=list(workflows.keys()),
                            workflow_graph_nodes=len(
                                getattr(workflow_graph_obj, "graph", {}) or {}
                            ),
                        ),
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
                    _emit_meta_trace(
                        logger,
                        "startup readiness evaluation beginning",
                        workflow_count=len(workflows),
                        planner_available=planner_cls is not None,
                        bootstrap_mode=bootstrap_mode,
                    )
                    print(
                        "[META-TRACE] startup readiness evaluation initiated; workflows=%d planner=%s bootstrap=%s"
                        % (
                            len(workflows),
                            getattr(planner_cls, "__name__", str(planner_cls)),
                            bootstrap_mode,
                        ),
                        flush=True,
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
                        print(
                            "[META-TRACE] prelaunch ROI cycles completed; ready=%s backoff=%s"
                            % (ready_to_launch, roi_backoff_triggered),
                            flush=True,
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
                        print(
                            "[META-TRACE] prelaunch ROI cycle finished; ready=%s backoff=%s error=%s"
                            % (ready_to_launch, roi_backoff_triggered, readiness_error),
                            flush=True,
                        )
                        _emit_meta_trace(
                            logger,
                            "prelaunch ROI cycle invocation finished",
                            ready_to_launch=ready_to_launch,
                            roi_backoff=roi_backoff_triggered,
                            readiness_error=readiness_error,
                        )
    
                    if (
                        not ready_to_launch
                        and bootstrap_mode
                        and not roi_backoff_triggered
                        and planner_cls is not None
                        and workflows
                    ):
                        logger.info(
                            "✅ bootstrap mode overriding diminishing returns gate; ROI baseline unavailable",
                            extra=log_record(
                                event="startup-bootstrap-diminishing-bypass",
                                workflow_count=len(workflows),
                            ),
                        )
                        ready_to_launch = True
    
                    if not ready_to_launch:
                        failure_reasons: list[str] = []
                        if not workflows:
                            failure_reasons.append("no workflows discovered")
                        if planner_cls is None:
                            failure_reasons.append("MetaWorkflowPlanner unavailable")
                        if readiness_error:
                            failure_reasons.append(readiness_error)
                        elif roi_backoff_triggered:
                            failure_reasons.append("ROI backoff triggered before launch")
                        else:
                            failure_reasons.append("ROI gate not satisfied")
    
                        logger.error(
                            "❌ sandbox readiness failed; aborting launch: %s",
                            "; ".join(failure_reasons),
                            extra=log_record(
                                event="startup-readiness-failed",
                                failure_reasons=failure_reasons,
                                planner_available=planner_cls is not None,
                                workflow_count=len(workflows),
                                roi_backoff=roi_backoff_triggered,
                            ),
                        )
                        sys.exit(1)
    
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
                    _emit_meta_trace(
                        logger,
                        "startup readiness evaluation complete",
                        workflow_count=len(workflows),
                        ready_to_launch=ready_to_launch,
                        roi_backoff=roi_backoff_triggered,
                        planner_available=planner_cls is not None,
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
                    failure_reasons = []
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
                            "🚦 meta planning launch block reached; beginning verbose instrumentation",
                            extra=log_record(
                                event="meta-planning-launch-block-entry",
                                workflow_count=len(workflows),
                                planner_cls=str(planner_cls),
                                roi_backoff=roi_backoff_triggered,
                                readiness_error=readiness_error,
                                bootstrap_mode=bootstrap_mode,
                                correlation_id=cid,
                            ),
                        )
                        logger.info(
                            "meta planning loop prerequisites verified",
                            extra=log_record(
                                event="meta-planning-loop-prereq",
                                workflow_count=len(workflows),
                                planner_available=planner_cls is not None,
                            ),
                        )
                        if not workflows or planner_cls is None:
                            logger.error(
                                "❌ meta planning loop prerequisites missing; aborting start",
                                extra=log_record(
                                    event="meta-planning-loop-prereq-missing",
                                    workflow_count=len(workflows),
                                    planner_available=planner_cls is not None,
                                ),
                            )
                            sys.exit(1)
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
                                workflow_graph_type=type(workflow_graph_obj).__name__,
                                workflow_graph_is_graph=(
                                    getattr(workflow_graph_obj, "graph", None) is not None
                                ),
                                event_bus_type=type(shared_event_bus).__name__
                                if shared_event_bus is not None
                                else None,
                                event_bus_handlers=getattr(
                                    shared_event_bus, "listeners", None
                                ),
                            ),
                        )
                        logger.info(
                            "🛰️ verifying meta_planning module attributes prior to start_self_improvement_cycle()",
                            extra=log_record(
                                event="meta-planning-module-precheck",
                                module_dir=list(sorted(dir(meta_planning))),
                                has_reload_settings=hasattr(meta_planning, "reload_settings"),
                                has_self_improvement_cycle=hasattr(
                                    meta_planning, "start_self_improvement_cycle"
                                ),
                                callable_self_improvement_cycle=callable(
                                    getattr(meta_planning, "start_self_improvement_cycle", None)
                                ),
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
                                "🛰️ deep-dive meta planning bootstrap parameter snapshot",
                                extra=log_record(
                                    event="meta-planning-bootstrap-param-snapshot",
                                    workflow_keys=list(workflows.keys()),
                                    workflow_len=len(workflows),
                                    planner_cls=str(planner_cls),
                                    interval_seconds=interval,
                                    workflow_graph_type=type(workflow_graph_obj).__name__,
                                    workflow_graph_keys=list((workflow_graph_obj or {}).keys())
                                    if isinstance(workflow_graph_obj, Mapping)
                                    else None,
                                    event_bus_type=type(shared_event_bus).__name__
                                    if shared_event_bus is not None
                                    else None,
                                    event_bus_has_listeners=bool(
                                        getattr(shared_event_bus, "listeners", None)
                                    ),
                                ),
                            )
                            logger.info(
                                "🛰️ recording meta planning invocation parameters for traceability",
                                extra=log_record(
                                    event="meta-planning-bootstrap-args",
                                    workflow_ids=list(workflows.keys()),
                                    workflow_count=len(workflows),
                                    interval_seconds=interval,
                                    planner_cls=str(planner_cls),
                                    workflow_graph_repr=repr(workflow_graph_obj),
                                    event_bus_repr=repr(shared_event_bus),
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
                                    callable_object=getattr(
                                        meta_planning, "start_self_improvement_cycle", None
                                    ),
                                    callable_is_function=callable(
                                        getattr(
                                            meta_planning, "start_self_improvement_cycle", None
                                        )
                                    ),
                                ),
                            )
                            if not hasattr(meta_planning, "start_self_improvement_cycle"):
                                logger.error(
                                    "❌ start_self_improvement_cycle missing on meta_planning module",
                                    extra=log_record(
                                        event="meta-planning-missing-entrypoint",
                                        module_dir=list(dir(meta_planning)),
                                        planner_cls=str(planner_cls),
                                    ),
                                )
                                raise RuntimeError(
                                    "start_self_improvement_cycle missing on meta_planning"
                                )
                            if not callable(
                                getattr(meta_planning, "start_self_improvement_cycle", None)
                            ):
                                logger.error(
                                    "❌ start_self_improvement_cycle present but not callable",
                                    extra=log_record(
                                        event="meta-planning-entrypoint-not-callable",
                                        type_info=type(
                                            getattr(
                                                meta_planning, "start_self_improvement_cycle", None
                                            )
                                        ).__name__,
                                        planner_cls=str(planner_cls),
                                    ),
                                )
                                raise RuntimeError(
                                    "start_self_improvement_cycle is not callable"
                                )
                            logger.info(
                                "🛰️ start_self_improvement_cycle() callable confirmed; executing with detailed context",
                                extra=log_record(
                                    event="meta-planning-callable-confirmed",
                                    workflow_count=len(workflows),
                                    planner_cls=str(planner_cls),
                                    interval_seconds=interval,
                                    workflow_graph_snapshot=repr(workflow_graph_obj),
                                    event_bus_snapshot=repr(shared_event_bus),
                                    caller_module=__name__,
                                ),
                            )
                            thread = meta_planning.start_self_improvement_cycle(
                                workflows,
                                event_bus=shared_event_bus,
                                interval=interval,
                                workflow_graph=workflow_graph_obj,
                            )
                            logger.info(
                                "🛰️ start_self_improvement_cycle() invocation completed; capturing return object",
                                extra=log_record(
                                    event="meta-planning-post-invoke",
                                    returned_type=type(thread).__name__ if thread is not None else None,
                                    returned_is_thread=isinstance(thread, threading.Thread),
                                    returned_repr=repr(thread),
                                ),
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
                                    thread_target=getattr(thread, "_target", None),
                                    thread_args=getattr(thread, "_args", None),
                                    thread_kwargs=getattr(thread, "_kwargs", None),
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
                                    native_id=getattr(thread, "native_id", None),
                                    ident=getattr(thread, "ident", None),
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
                                    native_id=getattr(thread, "native_id", None),
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
                                    thread_ident=getattr(thread, "ident", None),
                                    thread_native_id=getattr(thread, "native_id", None),
                                    thread_target=getattr(thread, "_target", None),
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
                                    thread_ident=getattr(thread, "ident", None),
                                    native_id=getattr(thread, "native_id", None),
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
                                    native_id=getattr(thread, "native_id", None),
                                ),
                            )
                            logger.info(
                                "🛰️ meta planning loop thread start diagnostics captured",
                                extra=log_record(
                                    event="meta-planning-thread-start-diagnostics",
                                    thread_name=getattr(thread, "name", "unknown"),
                                    thread_ident=getattr(thread, "ident", None),
                                    native_id=getattr(thread, "native_id", None),
                                    daemon=getattr(thread, "daemon", None),
                                    alive=getattr(thread, "is_alive", lambda: False)(),
                                    target=getattr(thread, "_target", None),
                                    args=getattr(thread, "_args", None),
                                    kwargs=getattr(thread, "_kwargs", None),
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
                                "🟢 meta planning loop thread running",
                                extra=log_record(
                                    event="meta-planning-thread-running",
                                    thread_name=getattr(thread, "name", "unknown"),
                                    native_id=getattr(thread, "native_id", None),
                                    ident=getattr(thread, "ident", None),
                                ),
                            )
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

            except Exception as exc:
                logger.exception(
                    "Early startup failure before meta-trace instrumentation",
                    extra=log_record(
                        event="startup-pre-meta-trace-failure",
                        last_step=last_pre_meta_trace_step,
                        error_type=type(exc).__name__,
                    ),
                )
                print(
                    "[DEBUG] startup failed before META-TRACE logging; last_step=%s; error=%s"
                    % (last_pre_meta_trace_step, exc),
                    flush=True,
                )
                raise

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
        if roi_backoff_triggered or not ready_to_launch:
            if roi_backoff_triggered and "ROI backoff triggered before launch" not in failure_reasons:
                failure_reasons.append("ROI backoff triggered before launch")
            summary = "; ".join(failure_reasons) if failure_reasons else "launch conditions not met"
            logger.error(
                "sandbox launch blocked: %s",
                summary,
                extra=log_record(
                    event="sandbox-launch-blocked",
                    failure_reasons=failure_reasons,
                    roi_backoff=roi_backoff_triggered,
                    ready_to_launch=ready_to_launch,
                    correlation_id=cid,
                ),
            )
            sys.stderr.write(f"sandbox launch blocked: {summary}\n")
            sys.stderr.flush()
            sys.exit(3)
        print("[start_autonomous_sandbox] launching sandbox", flush=True)
        launch_sandbox()
        print("[start_autonomous_sandbox] sandbox exited", flush=True)
        logger.info("sandbox shutdown", extra=log_record(event="shutdown"))
    except KeyboardInterrupt:
        logger.warning(
            "sandbox interrupted; requesting shutdown",
            extra=log_record(
                event="sandbox-interrupt",
                bootstrap_alive=bootstrap_thread.is_alive()
                if bootstrap_thread
                else False,
                last_bootstrap_step=BOOTSTRAP_PROGRESS.get("last_step"),
            ),
        )
        if bootstrap_stop_event is not None:
            bootstrap_stop_event.set()
        if bootstrap_thread and bootstrap_thread.is_alive():
            bootstrap_thread.join(timeout=5)
        try:
            shutdown_autonomous_sandbox(timeout=5)
        except Exception:
            logger.exception(
                "shutdown after interrupt failed",
                extra=log_record(event="sandbox-interrupt-shutdown-error"),
            )
        raise SystemExit(130)
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
