"""Workflow execution logic."""

from __future__ import annotations

import logging
from typing import Any, Dict

from menace.core.evaluator import evaluate_roi
from menace.errors.exceptions import InputValidationError, MalformedInputError, MissingFieldError

logger = logging.getLogger(__name__)


def run_workflow(input: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single workflow in a deterministic, validated manner.

    Args:
        input: Workflow payload containing:
            - workflow_id: Unique identifier for the workflow.
            - payload: Dict containing ROI inputs.

    Returns:
        Dictionary with the exact schema: status, data, errors, meta.

    Raises:
        MissingFieldError: If required fields are missing or None.
        MalformedInputError: If input payload is not a dict or is malformed.
        InputValidationError: If workflow_id is invalid.
    """
    if not isinstance(input, dict):
        raise MalformedInputError(
            "Workflow input must be a dict",
            details={"received_type": type(input).__name__},
        )

    if "workflow_id" not in input or input.get("workflow_id") is None:
        raise MissingFieldError("workflow_id")
    workflow_id = input["workflow_id"]
    if not isinstance(workflow_id, str) or not workflow_id.strip():
        raise InputValidationError(
            "workflow_id must be a non-empty string",
            details={"workflow_id": workflow_id},
        )

    if "payload" not in input or input.get("payload") is None:
        raise MissingFieldError("payload")

    payload = input["payload"]
    if not isinstance(payload, dict):
        raise MalformedInputError(
            "payload must be a dict",
            details={"payload_type": type(payload).__name__},
        )

    logger.info("Running workflow", extra={"workflow_id": workflow_id})

    roi_result = evaluate_roi(payload)

    response = {
        "status": roi_result["status"],
        "data": {
            "workflow_id": workflow_id,
            "result": roi_result["data"],
        },
        "errors": roi_result["errors"],
        "meta": {
            "workflow_id": workflow_id,
            "inputs": roi_result["meta"].get("inputs", {}),
        },
    }

    logger.info("Workflow completed", extra={"workflow_id": workflow_id})

    return response
