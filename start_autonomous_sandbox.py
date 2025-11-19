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

from shared_event_bus import event_bus as shared_event_bus
from workflow_evolution_manager import WorkflowEvolutionManager
from self_improvement.workflow_discovery import discover_workflow_specs
from sandbox_orchestrator import SandboxOrchestrator

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


def _build_self_improvement_workflows(
    bootstrap_context: Mapping[str, Any],
    settings: SandboxSettings,
    workflow_evolver: WorkflowEvolutionManager,
    *,
    logger: logging.Logger,
    discovered_specs: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Callable[[], Any]]:
    """Return workflow callables derived from bootstrap and stored records."""

    workflows: dict[str, Callable[[], Any]] = {}

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

        workflows[f"preseeded_{name}"] = (lambda v=value: v)

    records = _load_workflow_records(settings, discovered_specs=discovered_specs)
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
            seq_str = (
                "-".join(str(step) for step in seq)
                if isinstance(seq, Sequence) and not isinstance(seq, (str, bytes))
                else str(seq)
            )
            workflows[workflow_id] = workflow_evolver.build_callable(seq_str)
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

    logger.info(
        "registered %d workflows for meta planning",
        len(workflows),
        extra=log_record(workflow_count=len(workflows)),
    )

    return workflows


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
    args = parser.parse_args(argv_list)

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
                workflows = _build_self_improvement_workflows(
                    bootstrap_context,
                    settings,
                    workflow_evolver,
                    logger=logger,
                    discovered_specs=discovered_specs,
                )
                try:
                    thread = meta_planning.start_self_improvement_cycle(
                        workflows,
                        event_bus=shared_event_bus,
                        interval=interval,
                    )
                except Exception:
                    logger.exception(
                        "failed to initialize meta planning loop; sandbox launch halted",
                        extra=log_record(event="meta-loop-error"),
                    )
                    sys.exit(1)

                thread.start()
                logger.info(
                    "meta planning loop thread started successfully",
                    extra=log_record(event="meta-planning-start"),
                )
                logger.info(
                    "preseeded bootstrap context in use; pipeline and manager are cached",
                    extra=log_record(event="bootstrap-preseed"),
                )

                orchestrator = SandboxOrchestrator(
                    workflows,
                    logger=logger,
                    loop_interval=float(os.getenv("GLOBAL_ORCHESTRATOR_INTERVAL", "30")),
                    diminishing_threshold=float(
                        os.getenv("GLOBAL_ROI_DIMINISHING_THRESHOLD", "0.01")
                    ),
                    patience=int(os.getenv("GLOBAL_ROI_PATIENCE", "3")),
                )
                orchestrator_thread = threading.Thread(
                    target=orchestrator.run,
                    name="sandbox-orchestrator",
                    daemon=True,
                )
                orchestrator_thread.start()
                logger.info(
                    "sandbox orchestrator started",
                    extra=log_record(event="orchestrator-start"),
                )
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
