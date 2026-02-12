"""CLI entry point for Menace workflow self-debugging via the sandbox runner."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
import importlib
import os
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
from menace_sandbox.workflow_run_summary import roi_weighted_order
from sandbox_runner import run_workflow_simulations
from sandbox_runner.environment import is_self_debugger_sandbox_import_failure, module_name_from_module_not_found
from sandbox_settings import SandboxSettings
from task_handoff_bot import WorkflowDB
from sandbox_results_logger import record_self_debug_metrics
from objective_guard import ObjectiveGuard, ObjectiveGuardViolation
try:
    from menace.self_coding_manager import ObjectiveApprovalPolicy
except Exception:  # pragma: no cover - flat layout fallback
    from self_coding_manager import ObjectiveApprovalPolicy
from self_coding_policy import (
    ensure_self_coding_unsafe_paths_env,
    evaluate_patch_promotion,
    get_patch_promotion_policy,
    is_self_coding_unsafe_path,
)
from dynamic_path_router import resolve_path

LOGGER = logging.getLogger(__name__)

ensure_self_coding_unsafe_paths_env()
DEFAULT_METRICS_SOURCE = "menace_workflow_self_debug"
_SELF_DEBUG_PAUSED = False
_CERTIFICATION_PATH = Path(resolve_path("sandbox_data")) / "self_improvement_certification.json"


def _load_certification_state() -> dict[str, object]:
    try:
        data = json.loads(_CERTIFICATION_PATH.read_text(encoding="utf-8"))
    except OSError:
        return {}
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _is_self_improvement_certified() -> bool:
    state = _load_certification_state()
    return bool(state.get("certified"))


def _enable_certified_controls() -> None:
    os.environ.setdefault("SANDBOX_RECURSIVE_ORPHANS", "1")
    os.environ.setdefault("SANDBOX_RECURSIVE_ISOLATED", "1")
    os.environ.setdefault("SANDBOX_DISCOVER_ISOLATED", "1")
    os.environ.setdefault("SANDBOX_AUTO_INCLUDE_ISOLATED", "1")


def _mark_self_improvement_certified(
    *,
    roi_delta_total: float,
    source_path: Path,
    correlation_id: str | None,
) -> None:
    if _is_self_improvement_certified():
        return
    _CERTIFICATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "certified": True,
        "certified_at": datetime.utcnow().isoformat(),
        "roi_delta_total": roi_delta_total,
        "source_path": str(source_path),
        "correlation_id": correlation_id,
    }
    try:
        _CERTIFICATION_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        LOGGER.exception("failed to persist self-improvement certification state")


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


def _load_failure_context(
    context_path: Path | None,
    context_id: str | None,
) -> Mapping[str, object] | None:
    if context_path is None:
        return None
    try:
        data = json.loads(context_path.read_text(encoding="utf-8"))
    except OSError:
        LOGGER.exception("failed to read failure context file")
        return None
    except json.JSONDecodeError:
        LOGGER.exception("failed to parse failure context file")
        return None
    if context_id and isinstance(data, Mapping):
        failures = data.get("failures")
        if isinstance(failures, Mapping) and context_id in failures:
            entry = failures.get(context_id)
            if isinstance(entry, Mapping):
                return {
                    "context_path": str(context_path),
                    "context_id": context_id,
                    "record": dict(entry),
                }
    if isinstance(data, Mapping):
        return {
            "context_path": str(context_path),
            "record": dict(data),
        }
    return None


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
    LOGGER.info(
        "cell division event: promoted self-debug patch",
        extra={"source_path": str(source_path)},
    )
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
    promoted_authority = _is_self_improvement_certified()
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
            roi_delta=roi_delta_total,
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
    validation = logged_validate(
        str(pipeline_result.get("patch_text") or ""),
        allow_new_files=promoted_authority,
        allow_deletes=promoted_authority,
    )
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
            roi_delta=roi_delta_total,
            retry_count=0,
            metadata={
                "reason": "validation_failed",
                **(snapshot_meta or {}),
            },
        )
        return False
    pipeline_result = dict(pipeline_result)
    pipeline_result["validation"] = validation
    applied = logged_apply(
        pipeline_result,
        source_path=source_path,
        repo_root=repo_root,
        allow_new_files=promoted_authority,
        allow_deletes=promoted_authority,
    )
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
            roi_delta=roi_delta_total,
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
        roi_delta=roi_delta_total,
        retry_count=0,
        metadata={
            "reason": "patch_applied",
            **(snapshot_meta or {}),
        },
    )
    if roi_delta_total > 0:
        _mark_self_improvement_certified(
            roi_delta_total=roi_delta_total,
            source_path=source_path,
            correlation_id=correlation_id,
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
    failure_context: Mapping[str, object] | None = None,
) -> int:
    certified = _is_self_improvement_certified()
    modules = discover_workflow_modules(repo_root, include_bots=certified)
    forced_module = os.getenv("MENACE_SELF_DEBUG_MODULE")
    if forced_module:
        modules = [forced_module]
    if not certified and not forced_module and modules:
        modules = [modules[0]]
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
    if certified:
        _enable_certified_controls()
    ordered_modules = roi_weighted_order(modules) if certified else modules
    modules = ordered_modules
    effective_dynamic = dynamic_workflows or certified
    if failure_context:
        LOGGER.info(
            "self-debug failure context loaded",
            extra={
                "context_path": failure_context.get("context_path"),
                "context_id": failure_context.get("context_id"),
            },
        )
    snapshot = freeze_cycle(
        inputs={
            "workflow_modules": ordered_modules,
            "workflow_db_path": str(workflow_db_path),
            "source_menace_id": source_menace_id,
            "dynamic_workflows": effective_dynamic,
            "certified": certified,
            "failure_context": failure_context,
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

    snapshot_meta = {
        "snapshot_id": snapshot.snapshot_id,
        "snapshot_path": str(snapshot.path),
    }
    if certified:
        LOGGER.info(
            "self-improvement certification active",
            extra={
                "workflow_count": len(modules),
                "dynamic_workflows": effective_dynamic,
            },
        )
    if forced_module:
        try:
            mod = importlib.import_module(forced_module)
            workflow_fn = getattr(mod, "main", None) or getattr(mod, "run", None)
            if not callable(workflow_fn):
                raise AttributeError(
                    f"module {forced_module} lacks main/run callable"
                )
            result = workflow_fn()
            if not result:
                raise RuntimeError(
                    f"workflow {forced_module} returned failure"
                )
        except Exception as exc:
            LOGGER.exception(
                "forced workflow failed; routing through mvp pipeline",
                extra={"workflow_module": forced_module},
            )
            patched = _handle_failure(
                exc,
                repo_root=repo_root,
                prior_roi=-1.0,
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
            dynamic_workflows=effective_dynamic,
            context_builder=context_builder,
        )
    except ModuleNotFoundError as exc:
        if not is_self_debugger_sandbox_import_failure(exc):
            raise
        missing_module = module_name_from_module_not_found(exc)
        primary_failure = (
            failure_context.get("record") if isinstance(failure_context, Mapping) else None
        )
        LOGGER.exception(
            "workflow fallback self-debug dependency/layout failure",
            extra={
                "classification": "dependency_layout_failure",
                "fallback_failure": "self_debugger_sandbox import failure",
                "missing_module": missing_module,
                "primary_failure": primary_failure,
            },
        )
        classification = "dependency_layout_failure"
        if metrics_logger:
            metrics_logger.log_metrics(
                "menace.self_debug.exit",
                attempts=0,
                classification={"status": classification},
                patch_validity=None,
                roi_delta=None,
                roi_delta_total=None,
                exit_reason="fallback_dependency_layout_failure",
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
                    "roi_delta_total": None,
                    "roi_delta": None,
                    "exit_reason": "fallback_dependency_layout_failure",
                    "primary_failure": primary_failure,
                    "fallback_failure": {
                        "type": "self_debugger_sandbox import failure",
                        "missing_module": missing_module,
                    },
                }
            )
        return 2
    except Exception as exc:
        LOGGER.exception("workflow simulation failed; routing through mvp pipeline")
        patched = _handle_failure(
            exc,
            repo_root=repo_root,
            prior_roi=-1.0,
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
    parser.add_argument(
        "--failure-context-path",
        type=Path,
        default=None,
        help="Path to persisted failure context payload.",
    )
    parser.add_argument(
        "--failure-context-id",
        default=None,
        help="Identifier for the failure context entry to load.",
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
        failure_context = _load_failure_context(
            args.failure_context_path, args.failure_context_id
        )
        return _run_self_debug(
            repo_root=args.repo_root.resolve(),
            workflow_db_path=workflow_db,
            source_menace_id=args.source_menace_id,
            dynamic_workflows=args.dynamic_workflows,
            metrics_logger=metrics_logger,
            correlation_id=correlation_id,
            metrics_source=args.metrics_source,
            failure_context=failure_context,
        )
    finally:
        metrics_logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
