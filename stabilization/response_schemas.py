from __future__ import annotations

from typing import Any, Mapping

from simple_validation import SimpleField, SimpleSchema, ValidationError, fields


class MvpResponseSchema(SimpleSchema):
    objective = fields.Str()
    constraints = fields.List(fields.Str())
    generated_code = fields.Str()
    execution_output = fields.Str()
    execution_error = fields.Str()
    evaluation_error = fields.Str()
    roi_score = fields.Float()
    started_at = fields.Str()
    finished_at = fields.Str()
    duration_ms = fields.Int()
    success = fields.Bool()


class ErrorResponseSchema(SimpleSchema):
    error = fields.Str(required=True)
    success = fields.Bool()


class PatchValidationSchema(SimpleSchema):
    valid = fields.Bool()
    flags = fields.List(fields.Str())
    context = SimpleField(type=dict)


class PatchApplySchema(SimpleSchema):
    passed = fields.Bool()
    patch_id = SimpleField(type=None)
    flags = fields.List(fields.Str())


class AutonomousPresetBatchSchema(SimpleSchema):
    presets = fields.List(SimpleField(type=dict))
    preset_source = fields.Str()
    actions = fields.List(fields.Str())


def _normalize_payload(
    payload: Mapping[str, Any] | None,
    schema: type[SimpleSchema],
    defaults: Mapping[str, Any],
    *,
    extra_key: str | None = None,
) -> dict[str, Any]:
    if payload is None or not isinstance(payload, Mapping):
        raw: dict[str, Any] = {}
    else:
        raw = dict(payload)

    prepared: dict[str, Any] = dict(defaults)
    declared_fields = schema._declared_fields
    for key in declared_fields:
        if key in raw:
            prepared[key] = raw[key]

    normalized = schema().load(prepared)

    if extra_key:
        extras = {key: value for key, value in raw.items() if key not in declared_fields}
        if extras:
            normalized[extra_key] = extras

    return normalized


def normalize_mvp_response(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    defaults = {
        "objective": "",
        "constraints": [],
        "generated_code": "",
        "execution_output": "",
        "execution_error": "",
        "evaluation_error": "",
        "roi_score": 0.0,
        "started_at": "",
        "finished_at": "",
        "duration_ms": 0,
        "success": False,
    }
    return _normalize_payload(payload, MvpResponseSchema, defaults)


def normalize_error_response(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    defaults = {"error": "unknown error", "success": False}
    return _normalize_payload(payload, ErrorResponseSchema, defaults)


def normalize_patch_validation(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    defaults = {"valid": False, "flags": [], "context": {}}
    return _normalize_payload(payload, PatchValidationSchema, defaults)


def normalize_patch_apply(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    defaults = {"passed": False, "patch_id": None, "flags": []}
    return _normalize_payload(payload, PatchApplySchema, defaults)


def normalize_preset_batch(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    defaults = {"presets": [], "preset_source": "", "actions": []}
    return _normalize_payload(payload, AutonomousPresetBatchSchema, defaults, extra_key="extra")


__all__ = [
    "ValidationError",
    "normalize_mvp_response",
    "normalize_error_response",
    "normalize_patch_validation",
    "normalize_patch_apply",
    "normalize_preset_batch",
]
