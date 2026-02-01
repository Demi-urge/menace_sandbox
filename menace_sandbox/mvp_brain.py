"""Orchestrate MVP stabilization steps with logging instrumentation."""

from __future__ import annotations

from numbers import Number
from typing import Any, Mapping

from error_ontology import classify_error
import mvp_evaluator
from menace_sandbox import patch_generator
from menace_sandbox.stabilization.logging_wrapper import wrap_with_logging
from menace_sandbox.stabilization.roi import compute_roi_delta

__all__ = ["run_mvp_pipeline"]


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _pick_prior_roi(payload: Mapping[str, Any]) -> Number | None:
    candidates = (
        payload.get("prior_roi"),
        payload.get("roi_before"),
        payload.get("roi_prior"),
        payload.get("previous_roi"),
        payload.get("roi_previous"),
    )
    for value in candidates:
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return value
    return None


def _error_payload(message: str, code: str) -> dict[str, Any]:
    return {"status": "error", "errors": [{"code": code, "message": message}]}


def _validate_menace_patch_text(patch_text: str) -> dict[str, Any]:
    """Validate Menace patch text with the Menace patch validator."""

    validation = patch_generator.validate_patch_text(patch_text)
    if validation.get("valid"):
        validation.setdefault("context", {})["format"] = "menace_patch"
    return validation


def run_mvp_pipeline(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Run MVP classification, patching, validation, and ROI evaluation."""

    source = _coerce_text(payload.get("source") or payload.get("source_code"))
    rules = payload.get("rules") or []
    stdout = _coerce_text(payload.get("stdout"))
    stderr = _coerce_text(payload.get("stderr"))
    error_text = _coerce_text(payload.get("error")) or stderr
    returncode = payload.get("returncode", 0)
    validate_syntax = payload.get("validate_syntax")
    prior_roi = _pick_prior_roi(payload)

    logged_classify = wrap_with_logging(
        classify_error, {"log_event_prefix": "mvp.brain.classify."}
    )
    logged_generate = wrap_with_logging(
        patch_generator.generate_patch, {"log_event_prefix": "mvp.brain.patch.generate."}
    )
    logged_validate = wrap_with_logging(
        _validate_menace_patch_text, {"log_event_prefix": "mvp.brain.patch.validate."}
    )
    logged_evaluate_roi = wrap_with_logging(
        mvp_evaluator.evaluate_roi, {"log_event_prefix": "mvp.brain.roi.evaluate."}
    )
    logged_compute_delta = wrap_with_logging(
        compute_roi_delta, {"log_event_prefix": "mvp.brain.roi.delta."}
    )

    classification: Mapping[str, Any] | None = None
    if error_text:
        classification = logged_classify(error_text)

    error_report = {
        "stderr": error_text,
        "returncode": returncode,
        "classification": dict(classification or {}),
    }

    if not rules:
        patch_payload = {
            "status": "error",
            "errors": [
                {
                    "code": "missing_rules",
                    "message": "Patch rules are required for patch generation.",
                }
            ],
            "meta": {"missing_rules": True},
        }
    elif source:
        patch_payload = logged_generate(
            source,
            error_report,
            rules,
            validate_syntax=validate_syntax,
        )
    else:
        patch_payload = _error_payload(
            "source is required for patch generation", "missing_source"
        )

    patch_data = {}
    patch_meta = {}
    patch_status = ""
    patch_errors: list[Any] = []
    if isinstance(patch_payload, Mapping):
        patch_status = str(patch_payload.get("status") or "")
        patch_data = patch_payload.get("data") if isinstance(patch_payload.get("data"), Mapping) else {}
        patch_meta = patch_payload.get("meta") if isinstance(patch_payload.get("meta"), Mapping) else {}
        patch_errors = list(patch_payload.get("errors") or [])

    patch_text = _coerce_text(patch_data.get("patch_text"))
    modified_source = _coerce_text(
        patch_data.get("modified_source") or patch_data.get("updated_source")
    )
    if patch_text:
        validation_payload = logged_validate(patch_text)
    else:
        validation_payload = {
            "valid": False,
            "flags": ["missing_patch_text"],
            "context": {},
        }

    roi_score = float(logged_evaluate_roi(stdout, stderr))
    if prior_roi is None:
        roi_delta = {
            "status": "error",
            "data": {"deltas": {}, "total": 0},
            "errors": [
                {
                    "code": "missing_prior_roi",
                    "message": "Prior ROI is missing or invalid.",
                }
            ],
            "meta": {
                "keys": [],
                "count": 0,
                "error_count": 1,
                "before_count": 0,
                "after_count": 0,
            },
        }
    else:
        roi_delta = logged_compute_delta({"roi": prior_roi}, {"roi": roi_score})

    return {
        "classification": dict(classification or {}),
        "patch_metadata": {
            "status": patch_status,
            "errors": patch_errors,
            "meta": patch_meta,
        },
        "validation_flags": list(validation_payload.get("flags", [])),
        "validation": validation_payload,
        "roi_score": roi_score,
        "roi_delta": roi_delta,
        "patch_text": patch_text,
        "modified_source": modified_source,
    }
