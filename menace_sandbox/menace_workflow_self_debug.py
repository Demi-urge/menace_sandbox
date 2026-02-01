"""CLI entry point for Menace workflow self-debugging via the sandbox runner."""

from __future__ import annotations

import argparse
from datetime import datetime
import logging
from pathlib import Path
import tempfile
import traceback
import uuid
from typing import Iterable, Mapping

from menace_sandbox.context_builder_util import create_context_builder
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
from menace_sandbox.workflow_run_state import (
    classify_error,
    discover_workflow_modules,
    get_run_store,
    seed_workflow_db,
)
from menace_sandbox.menace_self_debug_snapshot import (
    _snapshot_environment,
    _snapshot_sandbox_settings,
    freeze_cycle,
)
from sandbox_runner import run_workflow_simulations
from sandbox_settings import SandboxSettings
from task_handoff_bot import WorkflowDB
from sandbox_results_logger import record_self_debug_metrics
from self_coding_policy import evaluate_patch_promotion, get_patch_promotion_policy

LOGGER = logging.getLogger(__name__)
DEFAULT_METRICS_SOURCE = "menace_workflow_self_debug"


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
    modified_source = pipeline_result.get("modified_source") or pipeline_result.get(
        "updated_source"
    )
    if not isinstance(modified_source, str) or not modified_source:
        return False

    original_source = source_path.read_text(encoding="utf-8")
    temp_dir = Path(tempfile.mkdtemp(prefix="menace_self_debug_"))
    temp_path = temp_dir / source_path.name
    try:
        temp_path.write_text(original_source, encoding="utf-8")
        temp_path.write_text(modified_source, encoding="utf-8")
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
    prior_roi: float,
    snapshot_meta: Mapping[str, object] | None = None,
    metrics_logger: StabilizationLoggingWrapper | None = None,
    correlation_id: str | None = None,
    metrics_source: str = DEFAULT_METRICS_SOURCE,
) -> bool:
    error_text = "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__)
    )
    run_store = get_run_store()
    source, source_path = _extract_source_from_traceback(exc, repo_root=repo_root)
    classification = classify_error(error_text)
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
    roi_delta = pipeline_result.get("roi_delta") if isinstance(pipeline_result, Mapping) else None
    if source_path is None:
        if metrics_logger:
            metrics_logger.log_metrics(
                "menace.self_debug.exit",
                attempts=0,
                classification={"status": classification},
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
                    "source": metrics_source,
                    "attempts": 0,
                    "classification": {"status": classification},
                    "patch_validity": None,
                    "roi_delta_total": roi_delta_total,
                    "roi_delta": roi_delta,
                    "exit_reason": "missing_source_path",
                }
            )
        run_store.record_run(
            workflow_id="self_debug",
            error_classification=classification,
            patch_attempts=0,
            roi_delta=0.0,
            retry_count=0,
            metadata={
                "reason": "missing_source_path",
                **(snapshot_meta or {}),
            },
        )
        return False
    logged_validate = wrap_with_logging(
        _validate_menace_patch_text, {"log_event_prefix": "menace.self_debug.patch.validate."}
    )
    logged_apply = wrap_with_logging(
        _apply_pipeline_patch, {"log_event_prefix": "menace.self_debug.patch.apply."}
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
                classification={"status": classification},
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
                    "source": metrics_source,
                    "attempts": 1,
                    "classification": {"status": classification},
                    "patch_validity": patch_validity,
                    "roi_delta_total": roi_delta_total,
                    "roi_delta": roi_delta,
                    "exit_reason": "validation_failed",
                }
            )
        run_store.record_run(
            workflow_id="self_debug",
            error_classification=classification,
            patch_attempts=1,
            roi_delta=0.0,
            retry_count=0,
            metadata={
                "reason": "validation_failed",
                **(snapshot_meta or {}),
            },
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
                classification={"status": classification},
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
                    "source": metrics_source,
                    "attempts": 1,
                    "classification": {"status": classification},
                    "patch_validity": patch_validity,
                    "roi_delta_total": roi_delta_total,
                    "roi_delta": roi_delta,
                    "exit_reason": "patch_not_applied",
                }
            )
        run_store.record_run(
            workflow_id="self_debug",
            error_classification=classification,
            patch_attempts=1,
            roi_delta=0.0,
            retry_count=0,
            metadata={
                "reason": "patch_not_applied",
                **(snapshot_meta or {}),
            },
        )
        return False
    if metrics_logger:
        metrics_logger.log_metrics(
            "menace.self_debug.exit",
            attempts=1,
            classification={"status": classification},
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
                "source": metrics_source,
                "attempts": 1,
                "classification": {"status": classification},
                "patch_validity": patch_validity,
                "roi_delta_total": roi_delta_total,
                "roi_delta": roi_delta,
                "exit_reason": "patch_applied",
            }
        )
    run_store.record_run(
        workflow_id="self_debug",
        error_classification=classification,
        patch_attempts=1,
        roi_delta=0.0,
        retry_count=0,
        metadata={
            "reason": "patch_applied",
            **(snapshot_meta or {}),
        },
    )
    return applied


def _run_self_debug(
    *,
    repo_root: Path,
    workflow_db_path: Path,
    source_menace_id: str,
    dynamic_workflows: bool,
    metrics_logger: StabilizationLoggingWrapper | None = None,
    correlation_id: str | None = None,
    metrics_source: str = DEFAULT_METRICS_SOURCE,
) -> int:
    modules = discover_workflow_modules(repo_root)
    try:
        settings = SandboxSettings()
    except Exception as exc:
        LOGGER.exception("failed to load sandbox settings", extra={"error": str(exc)})
        if metrics_logger:
            metrics_logger.log_metrics(
                "menace.self_debug.exit",
                attempts=0,
                classification=None,
                patch_validity=None,
                roi_delta=None,
                roi_delta_total=None,
                exit_reason="settings_load_failed",
            )
        if correlation_id:
            record_self_debug_metrics(
                {
                    "ts": datetime.utcnow().isoformat(),
                    "correlation_id": correlation_id,
                    "source": metrics_source,
                    "attempts": 0,
                    "classification": None,
                    "patch_validity": None,
                    "roi_delta_total": None,
                    "roi_delta": None,
                    "exit_reason": "settings_load_failed",
                }
            )
        return 1
    settings_snapshot = _snapshot_sandbox_settings(settings)
    snapshot = freeze_cycle(
        inputs={
            "workflow_modules": modules,
            "workflow_db_path": str(workflow_db_path),
            "source_menace_id": source_menace_id,
            "dynamic_workflows": dynamic_workflows,
        },
        configs={
            "sandbox_settings": settings_snapshot,
            "environment": _snapshot_environment(
                (
                    "SANDBOX_DATA_DIR",
                    "MENACE_DATA_DIR",
                    "WORKFLOW_DB_PATH",
                    "MENACE_WORKFLOW_DB",
                    "MENACE_LIGHT_IMPORTS",
                )
            ),
        },
        metadata={
            "source": metrics_source,
            "repo_root": str(repo_root),
        },
    )
    frozen_inputs = snapshot.payload.get("inputs", {})
    if isinstance(frozen_inputs, Mapping):
        frozen_modules = frozen_inputs.get("workflow_modules")
        if isinstance(frozen_modules, list):
            modules = [str(module) for module in frozen_modules if module]
    if not modules:
        LOGGER.warning("no workflow modules discovered")
        if metrics_logger:
            metrics_logger.log_metrics(
                "menace.self_debug.exit",
                attempts=0,
                classification=None,
                patch_validity=None,
                roi_delta=None,
                roi_delta_total=None,
                exit_reason="no_workflow_modules",
            )
        if correlation_id:
            record_self_debug_metrics(
                {
                    "ts": datetime.utcnow().isoformat(),
                    "correlation_id": correlation_id,
                    "source": metrics_source,
                    "attempts": 0,
                    "classification": None,
                    "patch_validity": None,
                    "roi_delta_total": None,
                    "roi_delta": None,
                    "exit_reason": "no_workflow_modules",
                }
            )
        return 1

    preflight_source = "def _layer4_preflight() -> int:\n    return 1\n"
    preflight_rules = build_rules(
        "Layer-4 preflight error",
        source=preflight_source,
        rule_source="layer4_preflight",
    )
    preflight_payload = {
        "source_code": preflight_source,
        "rules": preflight_rules,
        "stderr": "Layer-4 preflight error",
        "error": "Layer-4 preflight error",
        "returncode": 1,
        "prior_roi": 1.0,
    }
    preflight_result = run_mvp_pipeline(preflight_payload)
    validation = (
        preflight_result.get("validation") if isinstance(preflight_result, Mapping) else None
    )
    validation_ready = isinstance(validation, Mapping) and "valid" in validation
    LOGGER.info(
        "layer-4 preflight MVP validation check",
        extra={
            "validation_ready": validation_ready,
            "validation_flags": validation.get("flags", []) if isinstance(validation, Mapping) else None,
        },
    )
    if not validation_ready:
        if metrics_logger:
            metrics_logger.log_metrics(
                "menace.self_debug.exit",
                attempts=0,
                classification=None,
                patch_validity=None,
                roi_delta=None,
                roi_delta_total=None,
                exit_reason="mvp_validation_unavailable",
            )
        if correlation_id:
            record_self_debug_metrics(
                {
                    "ts": datetime.utcnow().isoformat(),
                    "correlation_id": correlation_id,
                    "source": metrics_source,
                    "attempts": 0,
                    "classification": None,
                    "patch_validity": None,
                    "roi_delta_total": None,
                    "roi_delta": None,
                    "exit_reason": "mvp_validation_unavailable",
                }
            )
        return 1

    snapshot_meta = {
        "snapshot_id": snapshot.snapshot_id,
        "snapshot_path": str(snapshot.path),
    }
    workflow_db = WorkflowDB(workflow_db_path)
    seeded = seed_workflow_db(
        modules, workflow_db=workflow_db, source_menace_id=source_menace_id
    )
    LOGGER.info("seeded workflows", extra={"count": len(seeded)})

    context_builder = create_context_builder(bootstrap_safe=True)
    try:
        logged_run = wrap_with_logging(
            run_workflow_simulations,
            {"log_event_prefix": "menace.self_debug.run."},
        )
        logged_run(
            workflows_db=str(workflow_db_path),
            env_presets=None,
            dynamic_workflows=dynamic_workflows,
            context_builder=context_builder,
        )
    except Exception as exc:
        LOGGER.exception("workflow simulation failed; routing through mvp pipeline")
        patched = _handle_failure(
            exc,
            repo_root=repo_root,
            prior_roi=0.0,
            snapshot_meta=snapshot_meta,
            metrics_logger=metrics_logger,
            correlation_id=correlation_id,
            metrics_source=metrics_source,
        )
        if patched:
            LOGGER.info("applied MVP patch for workflow failure")
            return 0
        return 1
    if metrics_logger:
        metrics_logger.log_metrics(
            "menace.self_debug.exit",
            attempts=0,
            classification=None,
            patch_validity=None,
            roi_delta=None,
            roi_delta_total=None,
            exit_reason="workflow_run_success",
        )
    if correlation_id:
        record_self_debug_metrics(
            {
                "ts": datetime.utcnow().isoformat(),
                "correlation_id": correlation_id,
                "source": metrics_source,
                "attempts": 0,
                "classification": None,
                "patch_validity": None,
                "roi_delta_total": None,
                "roi_delta": None,
                "exit_reason": "workflow_run_success",
            }
        )
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
        help="Workflow DB path (defaults to SandboxSettings.workflows_db).",
    )
    parser.add_argument(
        "--source-menace-id",
        default="menace_self_debug",
        help="Menace ID to store alongside seeded workflows.",
    )
    parser.add_argument(
        "--dynamic-workflows",
        action="store_true",
        help="Enable dynamic workflow generation for module groups.",
    )
    parser.add_argument(
        "--metrics-source",
        default=DEFAULT_METRICS_SOURCE,
        help="Metrics source tag for self-debug logging.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    workflow_db = args.workflow_db
    if workflow_db is None:
        workflow_db = Path(SandboxSettings().workflows_db)

    correlation_id = f"workflow-self-debug-{uuid.uuid4()}"
    LOGGER.info(
        "workflow self-debug starting",
        extra={
            "metrics_source": args.metrics_source,
            "workflow_db": str(workflow_db),
            "dynamic_workflows": args.dynamic_workflows,
        },
    )
    metrics_logger = StabilizationLoggingWrapper.start(
        correlation_id=correlation_id, source=args.metrics_source
    )
    try:
        return _run_self_debug(
            repo_root=args.repo_root.resolve(),
            workflow_db_path=workflow_db,
            source_menace_id=args.source_menace_id,
            dynamic_workflows=args.dynamic_workflows,
            metrics_logger=metrics_logger,
            correlation_id=correlation_id,
            metrics_source=args.metrics_source,
        )
    finally:
        metrics_logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
