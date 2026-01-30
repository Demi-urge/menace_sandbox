"""Workflow execution logic."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Mapping, Tuple

from menace.errors.exceptions import WorkflowExecutionError, WorkflowValidationError

SUPPORTED_STEP_TYPES = {"set_field", "copy_field", "merge_payload"}
MAX_STEPS = 100


def run_workflow(input: Dict[str, Any]) -> Dict[str, Any]:
    """Run a workflow with explicit schema validation and deterministic step execution.

    Workflow schema (all fields validated strictly):
        - workflow_id (str, required): unique identifier.
        - steps (list[dict], required): ordered list of step definitions.
        - payload (dict, required): mutable payload operated on by steps.
        - meta (dict, optional): additional metadata to echo in the response.

    Supported step types and behavior:
        - set_field: {"type": "set_field", "field": str, "value": Any}
          Sets payload[field] = value.
        - copy_field: {"type": "copy_field", "source": str, "target": str}
          Copies payload[source] to payload[target]. If source is missing, the step fails.
        - merge_payload: {"type": "merge_payload", "data": dict}
          Shallow merges payload with data, overwriting existing keys.

    Returns:
        Dictionary with the exact schema: status, data, errors, meta. The data object is
        always present and contains workflow_id, payload, and steps describing per-step
        outcomes. If any step fails, status is "partial_failure" and errors contains
        structured error entries while still returning the final payload.

    Raises:
        WorkflowValidationError: for malformed workflow definitions (schema or fields).
        WorkflowExecutionError: for unexpected execution failures.
    """

    workflow_id, steps, payload, meta = _validate_workflow_definition(input)

    payload_state = deepcopy(payload)
    step_results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    try:
        for index, step in enumerate(steps):
            step_type = step["type"]
            if step_type == "set_field":
                payload_state[step["field"]] = step["value"]
                step_results.append(_step_result(index, step_type, "success"))
                continue

            if step_type == "copy_field":
                source = step["source"]
                if source not in payload_state:
                    error = _step_error(
                        code="missing_source_field",
                        message="copy_field source does not exist in payload",
                        step_index=index,
                        step_type=step_type,
                        details={"source": source, "target": step["target"]},
                    )
                    errors.append(error)
                    step_results.append(_step_result(index, step_type, "error", error))
                    continue
                payload_state[step["target"]] = payload_state[source]
                step_results.append(_step_result(index, step_type, "success"))
                continue

            if step_type == "merge_payload":
                payload_state.update(step["data"])
                step_results.append(_step_result(index, step_type, "success"))
                continue

            raise WorkflowValidationError(
                "Unknown step type encountered after validation",
                details={"step_index": index, "step_type": step_type},
            )
    except WorkflowValidationError:
        raise
    except Exception as error:  # noqa: BLE001 - unexpected execution errors
        raise WorkflowExecutionError(
            "Workflow execution failed unexpectedly",
            details={"workflow_id": workflow_id, "error_type": type(error).__name__},
        ) from error

    status = "success" if not errors else "partial_failure"
    return {
        "status": status,
        "data": {
            "workflow_id": workflow_id,
            "payload": payload_state,
            "steps": step_results,
        },
        "errors": errors,
        "meta": {"workflow_id": workflow_id, **meta},
    }


def _validate_workflow_definition(
    input_data: Dict[str, Any],
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    if not isinstance(input_data, dict):
        raise WorkflowValidationError(
            "Workflow input must be a dictionary",
            details={"received_type": type(input_data).__name__},
        )

    allowed_keys = {"workflow_id", "steps", "payload", "meta"}
    unexpected_keys = set(input_data.keys()) - allowed_keys
    if unexpected_keys:
        raise WorkflowValidationError(
            "Workflow definition contains unsupported fields",
            details={"unexpected_keys": sorted(unexpected_keys)},
        )

    workflow_id = input_data.get("workflow_id")
    if not isinstance(workflow_id, str) or not workflow_id.strip():
        raise WorkflowValidationError(
            "workflow_id must be a non-empty string",
            details={"workflow_id": workflow_id},
        )

    steps = input_data.get("steps")
    if not isinstance(steps, list):
        raise WorkflowValidationError(
            "steps must be a list of step definitions",
            details={"workflow_id": workflow_id, "steps_type": type(steps).__name__},
        )
    if len(steps) > MAX_STEPS:
        raise WorkflowValidationError(
            "steps exceeds the maximum allowed length",
            details={"workflow_id": workflow_id, "max_steps": MAX_STEPS},
        )

    payload = input_data.get("payload")
    if not isinstance(payload, dict):
        raise WorkflowValidationError(
            "payload must be a dictionary",
            details={"workflow_id": workflow_id, "payload_type": type(payload).__name__},
        )

    meta = input_data.get("meta") or {}
    if not isinstance(meta, dict):
        raise WorkflowValidationError(
            "meta must be a dictionary when provided",
            details={"workflow_id": workflow_id, "meta_type": type(meta).__name__},
        )

    for index, step in enumerate(steps):
        _validate_step(step, workflow_id, index)

    return workflow_id, steps, payload, meta


def _validate_step(step: Mapping[str, Any], workflow_id: str, index: int) -> None:
    if not isinstance(step, Mapping):
        raise WorkflowValidationError(
            "step must be a mapping",
            details={"workflow_id": workflow_id, "step_index": index},
        )

    step_type = step.get("type")
    if step_type not in SUPPORTED_STEP_TYPES:
        raise WorkflowValidationError(
            "Unsupported step type",
            details={"workflow_id": workflow_id, "step_index": index, "step_type": step_type},
        )

    if step_type == "set_field":
        _validate_step_keys(step, {"type", "field", "value"}, workflow_id, index)
        field_value = step.get("field")
        if not isinstance(field_value, str) or not field_value.strip():
            raise WorkflowValidationError(
                "set_field requires a non-empty string field",
                details={"workflow_id": workflow_id, "step_index": index, "field": field_value},
            )
        return

    if step_type == "copy_field":
        _validate_step_keys(step, {"type", "source", "target"}, workflow_id, index)
        source = step.get("source")
        target = step.get("target")
        if not isinstance(source, str) or not source.strip():
            raise WorkflowValidationError(
                "copy_field requires a non-empty string source",
                details={"workflow_id": workflow_id, "step_index": index, "source": source},
            )
        if not isinstance(target, str) or not target.strip():
            raise WorkflowValidationError(
                "copy_field requires a non-empty string target",
                details={"workflow_id": workflow_id, "step_index": index, "target": target},
            )
        return

    if step_type == "merge_payload":
        _validate_step_keys(step, {"type", "data"}, workflow_id, index)
        data = step.get("data")
        if not isinstance(data, dict):
            raise WorkflowValidationError(
                "merge_payload requires a data dictionary",
                details={"workflow_id": workflow_id, "step_index": index, "data_type": type(data).__name__},
            )
        return


def _validate_step_keys(
    step: Mapping[str, Any],
    allowed_keys: set[str],
    workflow_id: str,
    index: int,
) -> None:
    unexpected = set(step.keys()) - allowed_keys
    if unexpected:
        raise WorkflowValidationError(
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
    step_index: int,
    step_type: str,
    details: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "code": code,
        "message": message,
        "step_index": step_index,
        "step_type": step_type,
        "details": details,
    }


def _step_result(
    index: int,
    step_type: str,
    status: str,
    error: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    result = {"index": index, "type": step_type, "status": status}
    if error:
        result["error"] = error
    return result
