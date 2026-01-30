"""Workflow execution logic."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Mapping

from menace.errors.exceptions import ValidationError, WorkflowError

LOGGER = logging.getLogger(__name__)
SUPPORTED_STEP_TYPES = {"set_field", "copy_field", "merge_payload"}
MAX_STEPS = 100


def run_workflow(input: dict[str, Any]) -> dict[str, Any]:
    """Run a workflow with explicit schema validation and deterministic execution.

    Args:
        input: Workflow definition with the following schema:
            workflow_id (str, required, non-empty)
            steps (list[dict], required, max length 100)
            payload (dict[str, Any], required, non-empty)
            meta (dict[str, Any], optional)

    Returns:
        dict[str, Any]: Dictionary with the schema:
            status (str): "ok" when all steps succeed, "error" when any step fails
            data (dict): workflow_id, payload, steps
            errors (list[dict]): structured errors captured during step execution
            metadata (dict): workflow_id plus any input meta, along with
                deterministic step-failure indicators:
                - partial_failure (bool): True when any step error exists
                - error_count (int): count of step-level errors
                - failed_steps (list[int]): indices of steps that failed
            meta (dict): Deprecated alias for ``metadata`` retained for backward
                compatibility. Both keys return identical payloads.

    Raises:
        ValidationError: If the input schema or step definitions are invalid.
        WorkflowError: If an unexpected execution failure occurs.
    """

    workflow_id, steps, payload, meta = _validate_workflow_definition(input)

    LOGGER.info(
        "workflow_runner.start",
        extra={"workflow_id": workflow_id, "steps": len(steps)},
    )

    payload_state = deepcopy(payload)
    step_results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    try:
        for index, step in enumerate(steps):
            step_type = step["type"]
            if step_type == "set_field":
                payload_state[step["field"]] = step["value"]
                step_results.append(_step_result(index, step_type, "ok"))
                continue

            if step_type == "copy_field":
                source = step["source"]
                if source not in payload_state:
                    error = _step_error(
                        code="missing_source_field",
                        message="copy_field source does not exist in payload",
                        workflow_id=workflow_id,
                        step_index=index,
                        step_type=step_type,
                        details={"source": source, "target": step["target"]},
                    )
                    errors.append(error)
                    step_results.append(_step_result(index, step_type, "error", error))
                    continue
                payload_state[step["target"]] = payload_state[source]
                step_results.append(_step_result(index, step_type, "ok"))
                continue

            if step_type == "merge_payload":
                payload_state.update(step["data"])
                step_results.append(_step_result(index, step_type, "ok"))
                continue

            raise ValidationError(
                "Unknown step type encountered after validation",
                details={"step_index": index, "step_type": step_type},
            )
    except ValidationError:
        raise
    except Exception as error:  # noqa: BLE001 - unexpected execution errors
        raise WorkflowError(
            "Workflow execution failed unexpectedly",
            details={"workflow_id": workflow_id, "error_type": type(error).__name__},
        ) from error
    finally:
        LOGGER.info(
            "workflow_runner.complete",
            extra={
                "workflow_id": workflow_id,
                "status": "error" if errors else "ok",
            },
        )

    partial_failure = bool(errors)
    status = "ok" if not partial_failure else "error"
    failed_steps = sorted({error.get("step_index") for error in errors if error.get("step_index") is not None})
    metadata = {
        "workflow_id": workflow_id,
        **meta,
        "partial_failure": partial_failure,
        "error_count": len(errors),
        "failed_steps": failed_steps,
    }
    return {
        "status": status,
        "data": {
            "workflow_id": workflow_id,
            "payload": payload_state,
            "steps": step_results,
        },
        "errors": errors,
        "metadata": metadata,
        "meta": metadata,
    }


def _validate_workflow_definition(
    input_data: dict[str, Any],
) -> tuple[str, list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Validate and normalize a workflow definition payload.

    Args:
        input_data (dict[str, Any]): Workflow definition mapping. Must be non-empty
            and contain non-``None`` values for required keys.

    Returns:
        tuple[str, list[dict[str, Any]], dict[str, Any], dict[str, Any]]: Ordered
        tuple of ``workflow_id``, validated ``steps``, ``payload``, and normalized
        ``meta`` dictionary.

    Raises:
        ValidationError: If the workflow schema is invalid, required fields are
            missing, values are ``None``, or field types are incorrect.

    Invariants:
        - Required keys are present and non-``None``.
        - ``payload`` contains no ``None`` values.
        - Validation is deterministic with respect to ``input_data``.
    """
    if not isinstance(input_data, dict) or not input_data:
        raise ValidationError(
            "Workflow input must be a non-empty dictionary",
            details={"received_type": type(input_data).__name__},
        )

    allowed_keys = {"workflow_id", "steps", "payload", "meta"}
    unexpected_keys = set(input_data.keys()) - allowed_keys
    if unexpected_keys:
        raise ValidationError(
            "Workflow definition contains unsupported fields",
            details={"unexpected_keys": sorted(unexpected_keys)},
        )

    missing_keys = {"workflow_id", "steps", "payload"} - set(input_data.keys())
    if missing_keys:
        raise ValidationError(
            "Workflow definition is missing required fields",
            details={"missing_keys": sorted(missing_keys)},
        )

    if input_data.get("workflow_id") is None:
        raise ValidationError(
            "workflow_id cannot be None",
            details={"workflow_id": input_data.get("workflow_id")},
        )

    if input_data.get("steps") is None:
        raise ValidationError(
            "steps cannot be None",
            details={"steps": input_data.get("steps")},
        )

    if input_data.get("payload") is None:
        raise ValidationError(
            "payload cannot be None",
            details={"payload": input_data.get("payload")},
        )

    workflow_id = input_data.get("workflow_id")
    if not isinstance(workflow_id, str) or not workflow_id.strip():
        raise ValidationError(
            "workflow_id must be a non-empty string",
            details={"workflow_id": workflow_id},
        )

    steps = input_data.get("steps")
    if not isinstance(steps, list):
        raise ValidationError(
            "steps must be a list of step definitions",
            details={"workflow_id": workflow_id, "steps_type": type(steps).__name__},
        )
    if len(steps) > MAX_STEPS:
        raise ValidationError(
            "steps exceeds the maximum allowed length",
            details={"workflow_id": workflow_id, "max_steps": MAX_STEPS},
        )

    payload = input_data.get("payload")
    if not isinstance(payload, dict) or not payload:
        raise ValidationError(
            "payload must be a non-empty dictionary",
            details={"workflow_id": workflow_id, "payload_type": type(payload).__name__},
        )
    _validate_payload_values(payload, workflow_id)

    meta = input_data.get("meta") or {}
    if not isinstance(meta, dict):
        raise ValidationError(
            "meta must be a dictionary when provided",
            details={"workflow_id": workflow_id, "meta_type": type(meta).__name__},
        )

    for index, step in enumerate(steps):
        _validate_step(step, workflow_id, index)

    return workflow_id, steps, payload, meta


def _validate_payload_values(payload: dict[str, Any], workflow_id: str) -> None:
    """Ensure payload values contain no ``None`` entries.

    Args:
        payload (dict[str, Any]): Workflow payload to inspect.
        workflow_id (str): Identifier used for validation error context.

    Returns:
        None: The payload is validated in-place (no mutation occurs).

    Raises:
        ValidationError: If any payload value is ``None``.

    Invariants:
        - No ``None`` values are allowed in payloads.
        - Validation is deterministic for a given payload.
    """
    null_keys = [key for key, value in payload.items() if value is None]
    if null_keys:
        raise ValidationError(
            "payload contains None values",
            details={"workflow_id": workflow_id, "null_keys": sorted(null_keys)},
        )


def _validate_step(step: Mapping[str, Any], workflow_id: str, index: int) -> None:
    """Validate a single workflow step definition.

    Args:
        step (Mapping[str, Any]): Step configuration mapping.
        workflow_id (str): Workflow identifier for error context.
        index (int): Step index in the workflow sequence.

    Returns:
        None: The step is validated without mutation.

    Raises:
        ValidationError: If the step is not a mapping, contains unsupported
            fields, has invalid types, or violates step-specific invariants.

    Invariants:
        - ``step`` must be a mapping with a supported ``type``.
        - Required step keys are present and non-``None``.
        - Validation is deterministic and order-independent for a given step.
    """
    if not isinstance(step, Mapping):
        raise ValidationError(
            "step must be a mapping",
            details={"workflow_id": workflow_id, "step_index": index},
        )

    step_type = step.get("type")
    if step_type not in SUPPORTED_STEP_TYPES:
        raise ValidationError(
            "Unsupported step type",
            details={"workflow_id": workflow_id, "step_index": index, "step_type": step_type},
        )

    if step_type == "set_field":
        _validate_step_keys(step, {"type", "field", "value"}, workflow_id, index)
        field_value = step.get("field")
        if not isinstance(field_value, str) or not field_value.strip():
            raise ValidationError(
                "set_field requires a non-empty string field",
                details={"workflow_id": workflow_id, "step_index": index, "field": field_value},
            )
        if step.get("value") is None:
            raise ValidationError(
                "set_field requires a non-null value",
                details={"workflow_id": workflow_id, "step_index": index, "field": field_value},
            )
        return

    if step_type == "copy_field":
        _validate_step_keys(step, {"type", "source", "target"}, workflow_id, index)
        source = step.get("source")
        target = step.get("target")
        if not isinstance(source, str) or not source.strip():
            raise ValidationError(
                "copy_field requires a non-empty string source",
                details={"workflow_id": workflow_id, "step_index": index, "source": source},
            )
        if not isinstance(target, str) or not target.strip():
            raise ValidationError(
                "copy_field requires a non-empty string target",
                details={"workflow_id": workflow_id, "step_index": index, "target": target},
            )
        return

    if step_type == "merge_payload":
        _validate_step_keys(step, {"type", "data"}, workflow_id, index)
        data = step.get("data")
        if data is None:
            raise ValidationError(
                "merge_payload requires non-null data",
                details={"workflow_id": workflow_id, "step_index": index},
            )
        if not isinstance(data, dict):
            raise ValidationError(
                "merge_payload requires a data dictionary",
                details={"workflow_id": workflow_id, "step_index": index, "data_type": type(data).__name__},
            )
        null_keys = [key for key, value in data.items() if value is None]
        if null_keys:
            raise ValidationError(
                "merge_payload data contains None values",
                details={"workflow_id": workflow_id, "step_index": index, "null_keys": sorted(null_keys)},
            )
        return


def _validate_step_keys(
    step: Mapping[str, Any],
    allowed_keys: set[str],
    workflow_id: str,
    index: int,
) -> None:
    """Verify that a step contains only allowed keys.

    Args:
        step (Mapping[str, Any]): Step payload to validate.
        allowed_keys (set[str]): Allowed keys for the step.
        workflow_id (str): Workflow identifier for error context.
        index (int): Step index for error context.

    Returns:
        None: The step is validated without mutation.

    Raises:
        ValidationError: If unexpected keys are present in the step.

    Invariants:
        - No unexpected keys are permitted for a given step type.
        - Validation is deterministic for identical inputs.
    """
    unexpected = set(step.keys()) - allowed_keys
    if unexpected:
        raise ValidationError(
            "Step contains unsupported fields",
            details={
                "workflow_id": workflow_id,
                "step_index": index,
                "unexpected_keys": sorted(unexpected),
            },
        )


def _step_error(
    code: str,
    message: str,
    workflow_id: str,
    step_index: int,
    step_type: str,
    details: dict[str, Any],
) -> dict[str, Any]:
    """Create a structured error payload for a workflow step.

    Args:
        code (str): Machine-readable error code.
        message (str): Human-readable error message.
        workflow_id (str): Workflow identifier.
        step_index (int): Index of the step that failed.
        step_type (str): Step type identifier.
        details (dict[str, Any]): Structured error details; must not be ``None``.

    Returns:
        dict[str, Any]: Deterministic error payload with code, message, and
        contextual metadata.

    Raises:
        None: This function does not raise.

    Invariants:
        - Returned payload has stable keys for a given input.
        - No ``None`` values are injected by this helper.
    """
    return {
        "code": code,
        "message": message,
        "workflow_id": workflow_id,
        "step_index": step_index,
        "step_type": step_type,
        "details": details,
    }


def _step_result(
    index: int,
    step_type: str,
    status: str,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic step result payload.

    Args:
        index (int): Step index in the workflow.
        step_type (str): Step type identifier.
        status (str): Result status label (e.g., ``"ok"`` or ``"error"``).
        error (dict[str, Any] | None): Optional error payload; when provided it
            must be a mapping.

    Returns:
        dict[str, Any]: Step result payload with optional error details.

    Raises:
        None: This function does not raise.

    Invariants:
        - Output keys are stable for identical inputs.
        - ``error`` is included only when provided.
    """
    result = {"index": index, "type": step_type, "status": status}
    if error:
        result["error"] = error
    return result
