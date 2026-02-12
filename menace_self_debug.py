from __future__ import annotations

"""Entry point for self-debugging core Menace workflows."""

import argparse
from dataclasses import dataclass
from datetime import datetime
import logging
import os
from pathlib import Path
import traceback
import uuid
from typing import Iterable, Mapping

from config_loader import load_config
from config_discovery import ConfigDiscovery
from default_config_manager import DefaultConfigManager
from menace_sandbox.mvp_brain import run_mvp_pipeline
from menace_sandbox.sandbox_rule_builder import build_rules
from menace_sandbox.stabilization.logging_wrapper import (
    StabilizationLoggingWrapper,
    wrap_with_logging,
)
from menace_sandbox.stabilization.patch_validator import (
    PatchValidationLimits,
    validate_patch_text,
)
from menace_sandbox.stabilization.roi import extract_roi_delta_total
from sandbox_results_logger import record_self_debug_metrics
from self_coding_policy import (
    ensure_self_coding_unsafe_paths_env,
    evaluate_patch_promotion,
    get_patch_promotion_policy,
    is_self_coding_unsafe_path,
)
from self_improvement.workflow_discovery import discover_workflow_specs
from objective_guard import ObjectiveGuard, ObjectiveGuardViolation
try:
    from menace.self_coding_manager import ObjectiveApprovalPolicy
except Exception:  # pragma: no cover - flat layout fallback
    from self_coding_manager import ObjectiveApprovalPolicy
from task_handoff_bot import WorkflowDB
from workflow_evolution_manager import _build_callable
from menace_sandbox.menace_self_debug_snapshot import (
    _snapshot_environment,
    freeze_cycle,
)

LOGGER = logging.getLogger(__name__)

ensure_self_coding_unsafe_paths_env()

_SELF_DEBUG_PAUSED = False


@dataclass(frozen=True)
class WorkflowCandidate:
    workflow_id: str
    sequence: tuple[str, ...]
    source: str


def _extract_source_from_traceback(
    exc: BaseException, *, repo_root: Path
) -> tuple[str, Path | None]:
    tb = traceback.TracebackException.from_exception(exc)
    for frame in reversed(tb.stack):
        candidate = Path(frame.filename)
        if not candidate.is_file():
            continue
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if repo_root in resolved.parents or resolved == repo_root:
            try:
                return resolved.read_text(encoding="utf-8"), resolved
            except OSError:
                return "", resolved
    return "", None


def _roi_delta_total(pipeline_result: Mapping[str, object]) -> float:
    roi_delta = pipeline_result.get("roi_delta")
    total = extract_roi_delta_total(roi_delta if isinstance(roi_delta, Mapping) else None)
    return float(total) if total is not None else 0.0


def _format_promotion_reasons(reasons: Iterable[str]) -> list[str]:
    mapping = {
        "roi_delta_invalid": "ROI delta ≤ 0",
        "roi_delta_not_positive": "ROI delta ≤ 0",
        "invalid_patch": "invalid patch",
        "unsafe_target": "unsafe file target",
    }
    return [mapping.get(reason, reason) for reason in reasons]


def _validate_menace_patch_text(
    patch_text: str,
    *,
    allow_new_files: bool = False,
    allow_deletes: bool = False,
) -> dict[str, object]:
    try:
        limits = PatchValidationLimits(
            allow_new_files=allow_new_files,
            allow_deletes=allow_deletes,
        )
        validation = validate_patch_text(patch_text, limits=limits)
    except Exception as exc:  # pragma: no cover - safety fallback
        validation = {
            "valid": False,
            "flags": ["validation_exception"],
            "context": {"error": str(exc)},
        }
    if not isinstance(validation, Mapping):
        return {
            "valid": False,
            "flags": ["validation_invalid_payload"],
            "context": {"payload_type": type(validation).__name__},
        }
    validation.setdefault("valid", False)
    validation.setdefault("flags", [])
    validation.setdefault("context", {})
    if validation.get("valid"):
        validation["context"]["format"] = "menace_patch"
    return dict(validation)


def _apply_pipeline_patch(
    pipeline_result: Mapping[str, object],
    *,
    source_path: Path,
    repo_root: Path,
    allow_new_files: bool = False,
    allow_deletes: bool = False,
) -> bool:
    global _SELF_DEBUG_PAUSED
    if _SELF_DEBUG_PAUSED:
        return False
    validation = pipeline_result.get("validation")
    if not isinstance(validation, Mapping) or not validation.get("valid"):
        return False
    patch_text = str(pipeline_result.get("patch_text") or "")
    if not patch_text:
        return False
    validation_result = _validate_menace_patch_text(
        patch_text,
        allow_new_files=allow_new_files,
        allow_deletes=allow_deletes,
    )
    if not validation_result.get("valid"):
        LOGGER.warning(
            "mvp patch failed menace validation",
            extra={
                "source_path": str(source_path),
                "rejection_reasons": ["invalid patch"],
            },
        )
        return False
    patch_paths = validation_result.get("context", {}).get("file_paths", [])
    if isinstance(patch_paths, list):
        for rel_path in patch_paths:
            if not isinstance(rel_path, str):
                continue
            if is_self_coding_unsafe_path(rel_path, repo_root=repo_root):
                LOGGER.info(
                    "mvp patch promotion rejected",
                    extra={
                        "source_path": str(source_path),
                        "rejection_reasons": ["unsafe file target"],
                        "unsafe_path": rel_path,
                    },
                )
                return False
    if is_self_coding_unsafe_path(source_path, repo_root=repo_root):
        LOGGER.info(
            "mvp patch promotion rejected",
            extra={
                "source_path": str(source_path),
                "rejection_reasons": ["unsafe file target"],
            },
        )
        return False
    policy = get_patch_promotion_policy(repo_root=repo_root)
    decision = evaluate_patch_promotion(
        policy=policy,
        roi_delta=pipeline_result.get("roi_delta")
        if isinstance(pipeline_result.get("roi_delta"), Mapping)
        else None,
        patch_validation=validation_result,
        source_path=source_path,
    )
    if not decision.allowed:
        LOGGER.info(
            "mvp patch promotion rejected",
            extra={
                "source_path": str(source_path),
                "rejection_reasons": _format_promotion_reasons(decision.reasons),
                "roi_delta_total": (
                    float(decision.roi_delta_total)
                    if decision.roi_delta_total is not None
                    else None
                ),
            },
        )
        return False
    approval_policy = ObjectiveApprovalPolicy(repo_root=repo_root)
    manual_token = os.getenv("MENACE_MANUAL_APPROVAL_TOKEN", "").strip() or None
    if not approval_policy.approve(source_path, manual_approval_token=manual_token):
        LOGGER.info(
            "mvp patch promotion rejected",
            extra={
                "source_path": str(source_path),
                "rejection_reasons": ["manual approval missing"],
                "classification": "objective_adjacent",
            },
        )
        return False

    modified_source = pipeline_result.get("modified_source") or pipeline_result.get(
        "updated_source"
    )
    if not isinstance(modified_source, str) or not modified_source:
        return False

    try:
        ObjectiveGuard(repo_root=repo_root).assert_integrity()
    except ObjectiveGuardViolation as exc:
        if getattr(exc, "reason", "") in {"objective_integrity_breach", "manifest_hash_mismatch"}:
            _SELF_DEBUG_PAUSED = True
            LOGGER.critical(
                "self-debug objective circuit breaker tripped",
                extra={
                    "event": "self_coding:objective_circuit_breaker_trip",
                    "severity": "critical",
                    "reason": getattr(exc, "reason", "objective_integrity_breach"),
                    "changed_files": list((getattr(exc, "details", {}) or {}).get("changed_files", [])),
                    "rollback_ok": False,
                },
            )
            return False
        raise

    original_source = source_path.read_text(encoding="utf-8")
    try:
        source_path.write_text(modified_source, encoding="utf-8")
    except Exception:
        LOGGER.exception(
            "failed to promote validated patch; rolling back",
            extra={"source_path": str(source_path)},
        )
        try:
            source_path.write_text(original_source, encoding="utf-8")
        except Exception:
            LOGGER.exception(
                "rollback failed after patch promotion error",
                extra={"source_path": str(source_path)},
            )
        return False
    return True


def _handle_failure(
    exc: BaseException,
    *,
    repo_root: Path,
    workflow_id: str,
    prior_roi: float,
    metrics_logger: StabilizationLoggingWrapper | None = None,
    correlation_id: str | None = None,
) -> bool:
    error_text = "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__)
    )
    source, source_path = _extract_source_from_traceback(exc, repo_root=repo_root)
    rules = build_rules(error_text, source=source)
    payload = {
        "source_code": source,
        "source_path": str(source_path) if source_path else None,
        "error": error_text,
        "stderr": error_text,
        "stdout": "",
        "returncode": 1,
        "rules": rules,
        "prior_roi": prior_roi,
    }
    logged_pipeline = wrap_with_logging(
        run_mvp_pipeline, {"log_event_prefix": "menace.self_debug.pipeline."}
    )
    pipeline_result = logged_pipeline(payload)
    roi_delta_total = _roi_delta_total(pipeline_result)
    roi_delta = (
        pipeline_result.get("roi_delta") if isinstance(pipeline_result, Mapping) else None
    )
    if source_path is None:
        if metrics_logger:
            metrics_logger.log_metrics(
                "menace.self_debug.exit",
                attempts=0,
                classification={"status": "missing_source_path"},
                patch_validity=None,
                roi_delta=roi_delta,
                roi_delta_total=roi_delta_total,
                exit_reason="missing_source_path",
            )
        if correlation_id:
            record_self_debug_metrics(
                {
                    "ts": datetime.utcnow().isoformat(),
                    "correlation_id": correlation_id,
                    "source": "menace_self_debug",
                    "attempts": 0,
                    "classification": {"status": "missing_source_path"},
                    "patch_validity": None,
                    "roi_delta_total": roi_delta_total,
                    "roi_delta": roi_delta,
                    "exit_reason": "missing_source_path",
                    "workflow_id": workflow_id,
                }
            )
        return False
    logged_validate = wrap_with_logging(
        _validate_menace_patch_text,
        {"log_event_prefix": "menace.self_debug.patch.validate."},
    )
    logged_apply = wrap_with_logging(
        _apply_pipeline_patch,
        {"log_event_prefix": "menace.self_debug.patch.apply."},
    )
    validation = logged_validate(str(pipeline_result.get("patch_text") or ""))
    patch_validity = validation if isinstance(validation, Mapping) else None
    if not validation.get("valid"):
        LOGGER.warning(
            "mvp patch validation failed; halting self-debug patch",
            extra={
                "flags": validation.get("flags", []),
                "rejection_reasons": ["invalid patch"],
            },
        )
        if metrics_logger:
            metrics_logger.log_metrics(
                "menace.self_debug.exit",
                attempts=1,
                classification={"status": "validation_failed"},
                patch_validity=patch_validity,
                roi_delta=roi_delta,
                roi_delta_total=roi_delta_total,
                exit_reason="validation_failed",
            )
        if correlation_id:
            record_self_debug_metrics(
                {
                    "ts": datetime.utcnow().isoformat(),
                    "correlation_id": correlation_id,
                    "source": "menace_self_debug",
                    "attempts": 1,
                    "classification": {"status": "validation_failed"},
                    "patch_validity": patch_validity,
                    "roi_delta_total": roi_delta_total,
                    "roi_delta": roi_delta,
                    "exit_reason": "validation_failed",
                    "workflow_id": workflow_id,
                }
            )
        return False
    pipeline_result = dict(pipeline_result)
    pipeline_result["validation"] = validation
    applied = logged_apply(pipeline_result, source_path=source_path, repo_root=repo_root)
    if not applied:
        LOGGER.info("mvp patch apply halted or failed")
        if metrics_logger:
            metrics_logger.log_metrics(
                "menace.self_debug.exit",
                attempts=1,
                classification={"status": "patch_not_applied"},
                patch_validity=patch_validity,
                roi_delta=roi_delta,
                roi_delta_total=roi_delta_total,
                exit_reason="patch_not_applied",
            )
        if correlation_id:
            record_self_debug_metrics(
                {
                    "ts": datetime.utcnow().isoformat(),
                    "correlation_id": correlation_id,
                    "source": "menace_self_debug",
                    "attempts": 1,
                    "classification": {"status": "patch_not_applied"},
                    "patch_validity": patch_validity,
                    "roi_delta_total": roi_delta_total,
                    "roi_delta": roi_delta,
                    "exit_reason": "patch_not_applied",
                    "workflow_id": workflow_id,
                }
            )
        return False
    if metrics_logger:
        metrics_logger.log_metrics(
            "menace.self_debug.exit",
            attempts=1,
            classification={"status": "patch_applied"},
            patch_validity=patch_validity,
            roi_delta=roi_delta,
            roi_delta_total=roi_delta_total,
            exit_reason="patch_applied",
        )
    if correlation_id:
        record_self_debug_metrics(
            {
                "ts": datetime.utcnow().isoformat(),
                "correlation_id": correlation_id,
                "source": "menace_self_debug",
                "attempts": 1,
                "classification": {"status": "patch_applied"},
                "patch_validity": patch_validity,
                "roi_delta_total": roi_delta_total,
                "roi_delta": roi_delta,
                "exit_reason": "patch_applied",
                "workflow_id": workflow_id,
            }
        )
    return applied


def _extract_sequence(record: Mapping[str, object]) -> tuple[str, ...]:
    for key in ("workflow", "task_sequence", "action_chains"):
        value = record.get(key)
        if isinstance(value, (list, tuple)):
            return tuple(str(item) for item in value if item)
        if isinstance(value, str) and value.strip():
            return tuple(part.strip() for part in value.split(",") if part.strip())
    steps = record.get("steps")
    if isinstance(steps, list):
        modules: list[str] = []
        for step in steps:
            if isinstance(step, Mapping):
                module = step.get("module")
                if module:
                    modules.append(str(module))
        if modules:
            return tuple(modules)
    return tuple()


def _load_registry_workflows(
    *,
    workflow_db_path: Path,
    scope: str,
    limit: int,
) -> list[WorkflowCandidate]:
    workflow_db = WorkflowDB(workflow_db_path)
    records = workflow_db.fetch_workflows(scope=scope, limit=limit)
    candidates: list[WorkflowCandidate] = []
    for record in records:
        sequence = _extract_sequence(record)
        workflow_id = str(record.get("workflow_id") or record.get("id") or "")
        if workflow_id and sequence:
            candidates.append(
                WorkflowCandidate(
                    workflow_id=workflow_id,
                    sequence=sequence,
                    source="registry",
                )
            )
    return candidates


def _load_discovered_workflows(
    *,
    repo_root: Path,
) -> list[WorkflowCandidate]:
    discovered = discover_workflow_specs(base_path=repo_root, logger=LOGGER)
    candidates: list[WorkflowCandidate] = []
    for spec in discovered:
        sequence = _extract_sequence(spec)
        workflow_id = str(
            (spec.get("metadata") or {}).get("workflow_id")
            or spec.get("workflow_id")
            or ""
        )
        if workflow_id and sequence:
            candidates.append(
                WorkflowCandidate(
                    workflow_id=workflow_id,
                    sequence=sequence,
                    source="discovery",
                )
            )
    return candidates


def _dedupe_candidates(
    candidates: Iterable[WorkflowCandidate],
) -> list[WorkflowCandidate]:
    seen: set[tuple[str, tuple[str, ...]]] = set()
    unique: list[WorkflowCandidate] = []
    for candidate in candidates:
        key = (candidate.workflow_id, candidate.sequence)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _load_configurations() -> None:
    DefaultConfigManager().apply_defaults()
    ConfigDiscovery().discover()
    try:
        load_config()
    except FileNotFoundError as exc:
        LOGGER.warning("Config missing; template created", exc_info=exc)
    except Exception:  # pragma: no cover - defensive logging
        LOGGER.exception("Config load failed")
        raise


def _resolve_workflow_db_path(workflow_db: Path | None) -> Path:
    if workflow_db is not None:
        return workflow_db
    env_path = None
    for env_key in ("WORKFLOW_DB_PATH", "MENACE_WORKFLOW_DB"):
        env_path = env_path or (Path(value) if (value := os.getenv(env_key)) else None)
    if env_path is not None:
        return env_path
    base_dir = os.getenv("MENACE_DATA_DIR") or os.getenv("SANDBOX_DATA_DIR") or "."
    return Path(base_dir) / "workflows.db"


def _run_self_debug(
    *,
    repo_root: Path,
    workflow_db_path: Path,
    scope: str,
    include_discovered: bool,
    max_workflows: int,
    max_failures: int,
    stop_on_failure: bool,
    metrics_logger: StabilizationLoggingWrapper | None = None,
    correlation_id: str | None = None,
) -> int:
    registry_candidates = _load_registry_workflows(
        workflow_db_path=workflow_db_path,
        scope=scope,
        limit=max_workflows,
    )
    discovered_candidates: list[WorkflowCandidate] = []
    if include_discovered:
        discovered_candidates = _load_discovered_workflows(repo_root=repo_root)

    candidates = _dedupe_candidates(registry_candidates + discovered_candidates)
    candidates = sorted(
        candidates,
        key=lambda candidate: (candidate.workflow_id, candidate.sequence, candidate.source),
    )
    snapshot = freeze_cycle(
        inputs={
            "candidates": [
                {
                    "workflow_id": candidate.workflow_id,
                    "sequence": list(candidate.sequence),
                    "source": candidate.source,
                }
                for candidate in candidates
            ],
            "workflow_db_path": str(workflow_db_path),
            "scope": scope,
            "include_discovered": include_discovered,
            "max_workflows": max_workflows,
            "max_failures": max_failures,
            "stop_on_failure": stop_on_failure,
        },
        configs={
            "environment": _snapshot_environment(
                (
                    "WORKFLOW_DB_PATH",
                    "MENACE_WORKFLOW_DB",
                    "MENACE_DATA_DIR",
                    "SANDBOX_DATA_DIR",
                )
            ),
        },
        metadata={
            "source": "menace_self_debug",
            "repo_root": str(repo_root),
        },
    )
    frozen_candidates = snapshot.payload.get("inputs", {}).get("candidates")
    if isinstance(frozen_candidates, list):
        candidates = []
        for entry in frozen_candidates:
            if not isinstance(entry, Mapping):
                continue
            workflow_id = str(entry.get("workflow_id") or "")
            sequence = entry.get("sequence") or []
            if isinstance(sequence, list):
                sequence_tuple = tuple(str(item) for item in sequence if item)
            else:
                sequence_tuple = tuple()
            source = str(entry.get("source") or "")
            if workflow_id and sequence_tuple:
                candidates.append(
                    WorkflowCandidate(
                        workflow_id=workflow_id,
                        sequence=sequence_tuple,
                        source=source or "snapshot",
                    )
                )
    if not candidates:
        LOGGER.warning("no workflows available to self-debug")
        if metrics_logger:
            metrics_logger.log_metrics(
                "menace.self_debug.exit",
                attempts=0,
                classification=None,
                patch_validity=None,
                roi_delta=None,
                roi_delta_total=None,
                exit_reason="no_workflows",
            )
        if correlation_id:
            record_self_debug_metrics(
                {
                    "ts": datetime.utcnow().isoformat(),
                    "correlation_id": correlation_id,
                    "source": "menace_self_debug",
                    "attempts": 0,
                    "classification": None,
                    "patch_validity": None,
                    "roi_delta_total": None,
                    "roi_delta": None,
                    "exit_reason": "no_workflows",
                }
            )
        return 1

    failures = 0
    for index, candidate in enumerate(candidates, start=1):
        if _SELF_DEBUG_PAUSED:
            LOGGER.critical("self-debug paused after objective circuit breaker")
            return 1
        if index > max_workflows:
            LOGGER.info("stop condition reached: max workflows")
            return 0
        try:
            workflow_fn = _build_callable("-".join(candidate.sequence))
            result = workflow_fn()
            if not result:
                raise RuntimeError(
                    f"workflow {candidate.workflow_id} returned failure"
                )
        except Exception as exc:
            LOGGER.exception(
                "workflow failed; routing through mvp pipeline",
                extra={"workflow_id": candidate.workflow_id, "source": candidate.source},
            )
            patched = _handle_failure(
                exc,
                repo_root=repo_root,
                workflow_id=candidate.workflow_id,
                prior_roi=0.0,
                metrics_logger=metrics_logger,
                correlation_id=correlation_id,
            )
            if patched:
                LOGGER.info("applied MVP patch for workflow failure")
                return 0
            failures += 1
            if failures >= max_failures:
                LOGGER.info("stop condition reached: max failures")
                return 1
            if stop_on_failure:
                LOGGER.info("stop condition reached: failure")
                return 1
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root for workflow discovery.",
    )
    parser.add_argument(
        "--workflow-db",
        type=Path,
        default=None,
        help="Workflow DB path (defaults to environment or ./workflows.db).",
    )
    parser.add_argument(
        "--scope",
        default="all",
        choices=("local", "global", "all"),
        help="WorkflowDB scope to use when loading workflows.",
    )
    parser.add_argument(
        "--include-discovered",
        action="store_true",
        help="Include auto-discovered workflow specs from the repository.",
    )
    parser.add_argument(
        "--max-workflows",
        type=int,
        default=20,
        help="Maximum number of workflows to execute.",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=1,
        help="Maximum number of failures before stopping.",
    )
    parser.add_argument(
        "--no-stop-on-failure",
        action="store_true",
        help="Continue after failures until max failures are reached.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    _load_configurations()

    repo_root = args.repo_root.resolve()
    workflow_db = _resolve_workflow_db_path(args.workflow_db)

    correlation_id = f"menace-self-debug-{uuid.uuid4()}"
    metrics_logger = StabilizationLoggingWrapper.start(
        correlation_id=correlation_id, source="menace_self_debug"
    )
    try:
        return _run_self_debug(
            repo_root=repo_root,
            workflow_db_path=workflow_db,
            scope=args.scope,
            include_discovered=args.include_discovered,
            max_workflows=args.max_workflows,
            max_failures=args.max_failures,
            stop_on_failure=not args.no_stop_on_failure,
            metrics_logger=metrics_logger,
            correlation_id=correlation_id,
        )
    finally:
        metrics_logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
