"""Synchronous workflow orchestrator."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from menace.core.workflow_runner import run_workflow
from menace.errors.exceptions import MenaceError, OrchestratorError

logger = logging.getLogger(__name__)


def _format_unexpected_error(exc: Exception) -> Dict[str, Any]:
    return OrchestratorError(
        "Unexpected orchestrator exception",
        details={"exception_type": type(exc).__name__, "message": str(exc)},
    ).to_error_dict()


def _normalize_input_data(input_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if input_data is None:
        logger.warning("Input data was None; defaulting to empty payload")
        return {}
    if not isinstance(input_data, dict):
        raise OrchestratorError(
            "input_data must be a dict",
            details={"received_type": type(input_data).__name__},
        )
    return input_data


def run_orchestrator(workflows: Sequence[Dict[str, Any]], input_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Run workflows in a single-pass deterministic execution.

    Args:
        workflows: Sequence of workflow payloads to execute.
        input_data: Shared input payload merged into workflow payloads.

    Returns:
        Dictionary with the exact schema: status, data, errors, meta.
    """
    logger.info(
        "Starting orchestrator run",
        extra={"workflow_count": len(workflows) if isinstance(workflows, Sequence) else 0},
    )

    try:
        normalized_input = _normalize_input_data(input_data)
    except MenaceError as exc:
        return {
            "status": "error",
            "data": {"results": []},
            "errors": [{"workflow_id": None, "index": None, "error": exc.to_error_dict()}],
            "meta": {"workflow_count": 0, "ok_count": 0, "error_count": 1},
        }

    if not isinstance(workflows, Sequence) or isinstance(workflows, (str, bytes)):
        error = OrchestratorError(
            "workflows must be a sequence of workflow payloads",
            details={"received_type": type(workflows).__name__},
        )
        return {
            "status": "error",
            "data": {"results": []},
            "errors": [{"workflow_id": None, "index": None, "error": error.to_error_dict()}],
            "meta": {"workflow_count": 0, "ok_count": 0, "error_count": 1},
        }

    if len(workflows) == 0:
        return {
            "status": "ok",
            "data": {"results": []},
            "errors": [],
            "meta": {"workflow_count": 0, "ok_count": 0, "error_count": 0},
        }

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    ok_count = 0
    error_count = 0

    try:
        for index, workflow in enumerate(workflows):
            workflow_id: Optional[str] = None
            try:
                if workflow is None:
                    raise OrchestratorError(
                        "Workflow definition cannot be None",
                        details={"index": index},
                    )
                if not isinstance(workflow, dict):
                    raise OrchestratorError(
                        "Each workflow must be a dict",
                        details={
                            "index": index,
                            "received_type": type(workflow).__name__,
                        },
                    )

                workflow_id = workflow.get("workflow_id")
                payload = workflow.get("payload")
                if payload is None:
                    workflow["payload"] = dict(normalized_input)
                elif not isinstance(payload, dict):
                    raise OrchestratorError(
                        "Workflow payload must be a dict",
                        details={
                            "index": index,
                            "workflow_id": workflow_id,
                            "payload_type": type(payload).__name__,
                        },
                    )
                else:
                    workflow["payload"] = {**normalized_input, **payload}

                result = run_workflow(workflow)
                results.append(result)
                if result.get("status") == "success":
                    ok_count += 1
                else:
                    error_count += 1
                    error_payload = (result.get("errors") or [{}])[0]
                    errors.append(
                        {
                            "workflow_id": workflow_id,
                            "index": index,
                            "error": error_payload,
                        }
                    )
            except MenaceError as exc:
                error_count += 1
                errors.append(
                    {
                        "workflow_id": workflow_id,
                        "index": index,
                        "error": exc.to_error_dict(),
                    }
                )
            except Exception as exc:  # unexpected exceptions
                error_count += 1
                errors.append(
                    {
                        "workflow_id": workflow_id,
                        "index": index,
                        "error": _format_unexpected_error(exc),
                    }
                )
    except Exception as exc:
        logger.exception("Unexpected orchestrator failure")
        orchestrator_error = OrchestratorError(
            "Orchestrator failed unexpectedly",
            details={"exception_type": type(exc).__name__, "message": str(exc)},
        )
        return {
            "status": "error",
            "data": {"results": results},
            "errors": [
                {
                    "workflow_id": None,
                    "index": None,
                    "error": orchestrator_error.to_error_dict(),
                }
            ],
            "meta": {"workflow_count": len(workflows), "ok_count": ok_count, "error_count": error_count + 1},
        }

    status = "ok"
    if error_count and ok_count:
        status = "partial_failure"
    elif error_count and not ok_count:
        status = "error"

    logger.info(
        "Orchestrator completed",
        extra={"status": status, "result_count": len(results), "error_count": error_count},
    )

    return {
        "status": status,
        "data": {"results": results},
        "errors": errors,
        "meta": {
            "workflow_count": len(workflows),
            "result_count": len(results),
            "ok_count": ok_count,
            "error_count": error_count,
        },
    }
