"""Workflow execution logic."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping

from menace.errors.exceptions import MenaceError, WorkflowError
from menace.infra.logging import get_logger, log_event

logger = get_logger(__name__)

WorkflowHandler = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def run_workflow(input: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single workflow in a deterministic, validated manner.

    Workflow input contract:
        - workflow_id (str, required): Unique identifier for the workflow.
        - payload (dict, required): Arbitrary workflow inputs passed to the handler.
        - handler (callable, required): Synchronous callable that accepts ``payload`` and
          returns a mapping result. Dynamic imports are intentionally disallowed; the
          caller must provide the handler directly.
        - meta (dict, optional): Extra metadata to include in the response ``meta``.

    Returns:
        Dictionary with the exact schema: status, data, errors, meta.
    """

    try:
        workflow_id, payload, handler, meta = _validate_input(input)
        log_event(
            logger=logger,
            level="info",
            event="workflow_started",
            status="running",
            metadata={"workflow_id": workflow_id},
        )

        result = handler(payload)
        if not isinstance(result, Mapping):
            raise WorkflowError(
                "Workflow handler must return a mapping",
                details={"workflow_id": workflow_id, "result_type": type(result).__name__},
            )

        response = {
            "status": "success",
            "data": {
                "workflow_id": workflow_id,
                "result": dict(result),
            },
            "errors": [],
            "meta": {
                "workflow_id": workflow_id,
                **meta,
            },
        }

        log_event(
            logger=logger,
            level="info",
            event="workflow_completed",
            status="success",
            metadata={"workflow_id": workflow_id},
        )
        return response
    except WorkflowError as error:
        return _handle_expected_failure(error)
    except MenaceError as error:
        wrapped = WorkflowError(
            error.message,
            details={"original_code": error.code, **error.details},
        )
        return _handle_expected_failure(wrapped)
    except Exception as error:  # noqa: BLE001 - unexpected exception handling
        log_event(
            logger=logger,
            level="error",
            event="workflow_unhandled_exception",
            status="error",
            metadata={"error_type": type(error).__name__},
            errors={"message": str(error)},
        )
        return {
            "status": "error",
            "data": None,
            "errors": [
                {
                    "code": "unhandled_exception",
                    "message": str(error),
                    "details": {},
                }
            ],
            "meta": {},
        }


def _validate_input(input_data: Dict[str, Any]) -> tuple[str, Dict[str, Any], WorkflowHandler, Dict[str, Any]]:
    """Validate the workflow input payload and return normalized values."""

    if not isinstance(input_data, dict):
        raise WorkflowError(
            "Workflow input must be a dict",
            details={"received_type": type(input_data).__name__},
        )

    workflow_id = input_data.get("workflow_id")
    if not isinstance(workflow_id, str) or not workflow_id.strip():
        raise WorkflowError(
            "workflow_id must be a non-empty string",
            details={"workflow_id": workflow_id},
        )

    payload = input_data.get("payload")
    if payload is None:
        raise WorkflowError(
            "payload is required for workflow execution",
            details={"workflow_id": workflow_id, "field": "payload"},
        )
    if not isinstance(payload, dict):
        raise WorkflowError(
            "payload must be a dict",
            details={"workflow_id": workflow_id, "payload_type": type(payload).__name__},
        )

    handler = input_data.get("handler")
    if handler is None:
        raise WorkflowError(
            "handler is required for workflow execution",
            details={"workflow_id": workflow_id, "field": "handler"},
        )
    if not callable(handler):
        raise WorkflowError(
            "handler must be callable",
            details={"workflow_id": workflow_id, "handler_type": type(handler).__name__},
        )

    meta = input_data.get("meta") or {}
    if not isinstance(meta, dict):
        raise WorkflowError(
            "meta must be a dict when provided",
            details={"workflow_id": workflow_id, "meta_type": type(meta).__name__},
        )

    return workflow_id, payload, handler, meta


def _handle_expected_failure(error: WorkflowError) -> Dict[str, Any]:
    """Return a structured response for expected workflow failures."""

    log_event(
        logger=logger,
        level="warning",
        event="workflow_expected_failure",
        status="error",
        errors=error.to_error_dict(),
        metadata={"workflow_id": error.details.get("workflow_id")},
    )
    return {
        "status": "error",
        "data": None,
        "errors": [error.to_error_dict()],
        "meta": {},
    }
