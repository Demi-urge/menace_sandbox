"""Synchronous workflow orchestrator."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from menace.core.workflow_runner import run_workflow
from menace.errors.exceptions import MalformedInputError, MenaceError

logger = logging.getLogger(__name__)


def _format_unexpected_error(exc: Exception) -> Dict[str, Any]:
    return {
        "message": str(exc),
        "code": "unexpected_error",
        "details": {
            "exception_type": type(exc).__name__,
        },
    }


def run_orchestrator(workflows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run workflows in a single-pass deterministic execution.

    Args:
        workflows: List of workflow payloads to execute.

    Returns:
        Dictionary with the exact schema: status, data, errors, meta.
    """
    logger.info("Starting orchestrator run", extra={"workflow_count": len(workflows) if isinstance(workflows, list) else 0})

    if not isinstance(workflows, list):
        error = MalformedInputError(
            "workflows must be a list of workflow payloads",
            details={"received_type": type(workflows).__name__},
        )
        return {
            "status": "error",
            "data": {"results": []},
            "errors": [{"workflow_id": None, "index": None, "error": error.to_error_dict()}],
            "meta": {"workflow_count": 0},
        }

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    try:
        for index, workflow in enumerate(workflows):
            workflow_id: Optional[str] = None
            try:
                if not isinstance(workflow, dict):
                    raise MalformedInputError(
                        "Each workflow must be a dict",
                        details={
                            "index": index,
                            "received_type": type(workflow).__name__,
                        },
                    )
                workflow_id = workflow.get("workflow_id")
                result = run_workflow(workflow)
                results.append(result)
            except MenaceError as exc:
                errors.append(
                    {
                        "workflow_id": workflow_id,
                        "index": index,
                        "error": exc.to_error_dict(),
                    }
                )
            except Exception as exc:  # unexpected exceptions
                errors.append(
                    {
                        "workflow_id": workflow_id,
                        "index": index,
                        "error": _format_unexpected_error(exc),
                    }
                )
    except Exception as exc:
        logger.exception("Unexpected orchestrator failure")
        return {
            "status": "error",
            "data": {"results": results},
            "errors": [{"workflow_id": None, "index": None, "error": _format_unexpected_error(exc)}],
            "meta": {"workflow_count": len(workflows)},
        }

    status = "ok"
    if errors and results:
        status = "partial_failure"
    elif errors and not results:
        status = "error"

    logger.info(
        "Orchestrator completed",
        extra={"status": status, "result_count": len(results), "error_count": len(errors)},
    )

    return {
        "status": status,
        "data": {"results": results},
        "errors": errors,
        "meta": {
            "workflow_count": len(workflows),
            "result_count": len(results),
            "error_count": len(errors),
        },
    }
