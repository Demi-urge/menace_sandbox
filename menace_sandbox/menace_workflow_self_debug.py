"""CLI entry point for Menace workflow self-debugging via the sandbox runner."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import traceback
from typing import Iterable, Mapping

from menace_sandbox.context_builder_util import create_context_builder
from menace_sandbox.mvp_brain import run_mvp_pipeline
from menace_sandbox import patch_generator
from menace_sandbox.sandbox_rule_builder import build_rules
from menace_sandbox.stabilization.logging_wrapper import wrap_with_logging
from menace_sandbox.workflow_run_state import (
    classify_error,
    discover_workflow_modules,
    get_run_store,
    seed_workflow_db,
)
from sandbox_runner import run_workflow_simulations
from sandbox_settings import SandboxSettings
from task_handoff_bot import WorkflowDB

LOGGER = logging.getLogger(__name__)


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
    if isinstance(roi_delta, Mapping):
        data = roi_delta.get("data")
        if isinstance(data, Mapping):
            try:
                return float(data.get("total", 0.0))
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def _validate_menace_patch_text(patch_text: str) -> dict[str, object]:
    try:
        patch_generator.validate_patch_text(patch_text)
    except Exception as exc:  # pragma: no cover - safety fallback
        return {
            "valid": False,
            "flags": ["validation_exception"],
            "context": {"error": str(exc)},
        }
    return {"valid": True, "flags": [], "context": {"format": "menace_patch"}}


def _apply_pipeline_patch(
    pipeline_result: Mapping[str, object], *, source_path: Path
) -> bool:
    validation = pipeline_result.get("validation")
    if not isinstance(validation, Mapping) or not validation.get("valid"):
        return False
    patch_text = str(pipeline_result.get("patch_text") or "")
    if not patch_text:
        return False
    validation_result = _validate_menace_patch_text(patch_text)
    if not validation_result.get("valid"):
        LOGGER.warning(
            "mvp patch failed menace validation",
            extra={"source_path": str(source_path)},
        )
        return False
    if _roi_delta_total(pipeline_result) <= 0:
        LOGGER.info(
            "mvp patch rejected due to non-positive roi delta",
            extra={"source_path": str(source_path)},
        )
        return False
    modified_source = pipeline_result.get("modified_source") or pipeline_result.get(
        "updated_source"
    )
    if not isinstance(modified_source, str) or not modified_source:
        return False
    source_path.write_text(modified_source, encoding="utf-8")
    return True


def _handle_failure(
    exc: BaseException,
    *,
    repo_root: Path,
    prior_roi: float,
) -> bool:
    error_text = "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__)
    )
    run_store = get_run_store()
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
    if source_path is None:
        run_store.record_run(
            workflow_id="self_debug",
            error_classification=classify_error(error_text),
            patch_attempts=0,
            roi_delta=0.0,
            retry_count=0,
            metadata={"reason": "missing_source_path"},
        )
        return False
    logged_validate = wrap_with_logging(
        _validate_menace_patch_text, {"log_event_prefix": "menace.self_debug.patch.validate."}
    )
    logged_apply = wrap_with_logging(
        _apply_pipeline_patch, {"log_event_prefix": "menace.self_debug.patch.apply."}
    )
    validation = logged_validate(str(pipeline_result.get("patch_text") or ""))
    if not validation.get("valid"):
        LOGGER.warning(
            "mvp patch validation failed; halting self-debug patch",
            extra={"flags": validation.get("flags", [])},
        )
        run_store.record_run(
            workflow_id="self_debug",
            error_classification=classify_error(error_text),
            patch_attempts=1,
            roi_delta=0.0,
            retry_count=0,
            metadata={"reason": "validation_failed"},
        )
        return False
    pipeline_result = dict(pipeline_result)
    pipeline_result["validation"] = validation
    applied = logged_apply(pipeline_result, source_path=source_path)
    if not applied:
        LOGGER.info("mvp patch apply halted or failed")
        run_store.record_run(
            workflow_id="self_debug",
            error_classification=classify_error(error_text),
            patch_attempts=1,
            roi_delta=0.0,
            retry_count=0,
            metadata={"reason": "patch_not_applied"},
        )
        return False
    run_store.record_run(
        workflow_id="self_debug",
        error_classification=classify_error(error_text),
        patch_attempts=1,
        roi_delta=0.0,
        retry_count=0,
        metadata={"reason": "patch_applied"},
    )
    return applied


def _run_self_debug(
    *,
    repo_root: Path,
    workflow_db_path: Path,
    source_menace_id: str,
    dynamic_workflows: bool,
) -> int:
    modules = discover_workflow_modules(repo_root)
    if not modules:
        LOGGER.warning("no workflow modules discovered")
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
            dynamic_workflows=dynamic_workflows,
            context_builder=context_builder,
        )
    except Exception as exc:
        LOGGER.exception("workflow simulation failed; routing through mvp pipeline")
        patched = _handle_failure(exc, repo_root=repo_root, prior_roi=0.0)
        if patched:
            LOGGER.info("applied MVP patch for workflow failure")
            return 0
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
    args = parser.parse_args(list(argv) if argv is not None else None)

    workflow_db = args.workflow_db
    if workflow_db is None:
        workflow_db = Path(SandboxSettings().workflows_db)

    return _run_self_debug(
        repo_root=args.repo_root.resolve(),
        workflow_db_path=workflow_db,
        source_menace_id=args.source_menace_id,
        dynamic_workflows=args.dynamic_workflows,
    )


if __name__ == "__main__":
    raise SystemExit(main())
