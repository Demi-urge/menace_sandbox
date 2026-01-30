"""Workflow execution logic."""

from __future__ import annotations

import logging
from typing import Any, Dict

from menace.core.evaluator import evaluate_roi
from menace.errors.exceptions import WorkflowDefinitionError

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
        WorkflowDefinitionError: If the workflow payload is malformed or incomplete.
    """
    if not isinstance(input, dict):
        raise WorkflowDefinitionError(
            "Workflow input must be a dict",
            details={"received_type": type(input).__name__},
        )

    if "workflow_id" not in input or input.get("workflow_id") is None:
        raise WorkflowDefinitionError(
            "workflow_id is required for workflow execution",
            details={"field": "workflow_id"},
        )
    workflow_id = input["workflow_id"]
    if not isinstance(workflow_id, str) or not workflow_id.strip():
        raise WorkflowDefinitionError(
            "workflow_id must be a non-empty string",
            details={"workflow_id": workflow_id},
        )

    if "payload" not in input or input.get("payload") is None:
        raise WorkflowDefinitionError(
            "payload is required for workflow execution",
            details={"field": "payload", "workflow_id": workflow_id},
        )

    payload = input["payload"]
    if not isinstance(payload, dict):
        raise WorkflowDefinitionError(
            "payload must be a dict",
            details={"payload_type": type(payload).__name__, "workflow_id": workflow_id},
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
