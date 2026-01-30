"""Synchronous workflow orchestrator."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from menace.core.evaluator import evaluate_roi
from menace.core.workflow_runner import run_workflow
from menace.errors.exceptions import ConfigError, EvaluatorError, MenaceError, OrchestratorError

logger = logging.getLogger(__name__)


def _error_dict(error: MenaceError) -> Dict[str, Any]:
    to_error_dict = getattr(error, "to_error_dict", None)
    if callable(to_error_dict):
        return to_error_dict()
    return {"message": error.message, "details": error.details}


def _format_unexpected_error(exc: Exception) -> Dict[str, Any]:
    return _error_dict(
        OrchestratorError(
            "Unexpected orchestrator exception",
            details={"exception_type": type(exc).__name__, "message": str(exc)},
        )
    )


def _normalize_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if config is None:
        raise ConfigError("config cannot be None", details={"received": None})
    if not isinstance(config, dict):
        raise ConfigError(
            "config must be a dict",
            details={"received_type": type(config).__name__},
        )
    if "input_data" not in config:
        raise ConfigError(
            "config missing required key 'input_data'",
            details={"required_keys": ["input_data"]},
        )
    return config


def _normalize_input_data(input_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if input_data is None:
        raise ConfigError(
            "config['input_data'] cannot be None",
            details={"received": None},
        )
    if not isinstance(input_data, dict):
        raise ConfigError(
            "config['input_data'] must be a dict",
            details={"received_type": type(input_data).__name__},
        )
    return input_data


def _resolve_evaluator_input(config: Dict[str, Any], results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if "evaluator_input" in config:
        evaluator_input = config.get("evaluator_input")
        if evaluator_input is None:
            raise ConfigError(
                "config['evaluator_input'] cannot be None when provided",
                details={"received": None},
            )
        if not isinstance(evaluator_input, dict):
            raise ConfigError(
                "config['evaluator_input'] must be a dict",
                details={"received_type": type(evaluator_input).__name__},
            )
        return evaluator_input

    for result in results:
        payload = (result.get("data") or {}).get("payload")
        if not isinstance(payload, dict):
            continue
        if payload.get("cost") is not None and payload.get("revenue") is not None:
            return {"cost": payload.get("cost"), "revenue": payload.get("revenue")}

    return None


def run_orchestrator(workflows: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    """Run workflows in a single-pass deterministic execution.

    Args:
        workflows: List of workflow payloads to execute. Each workflow is passed to
            ``run_workflow`` exactly once in order.
        config: Orchestrator configuration. Requires ``input_data`` for shared payload
            values and optionally accepts ``evaluator_input`` for ROI evaluation.

    Returns:
        Dictionary with the exact schema: status, data, errors, meta. Data contains
        workflow results and optional ROI evaluation output.

    Raises:
        None. This function handles ConfigError, WorkflowValidationError,
        WorkflowExecutionError, EvaluatorError, OrchestratorError, and unexpected
        Exception instances, returning structured error output instead of raising.
    """
    logger.info(
        "Starting orchestrator run",
        extra={"workflow_count": len(workflows) if isinstance(workflows, list) else 0},
    )

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    ok_count = 0
    error_count = 0
    evaluation_result: Optional[Dict[str, Any]] = None

    try:
        normalized_config = _normalize_config(config)
        normalized_input = _normalize_input_data(normalized_config.get("input_data"))

        if workflows is None:
            raise OrchestratorError(
                "workflows cannot be None",
                details={"received": None},
            )

        if not isinstance(workflows, list):
            raise OrchestratorError(
                "workflows must be a list of workflow payloads",
                details={"received_type": type(workflows).__name__},
            )

        if len(workflows) == 0:
            logger.info("No workflows provided; returning empty result set")
        else:
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
                            details={"index": index, "received_type": type(workflow).__name__},
                        )

                    workflow_id = workflow.get("workflow_id")
                    payload = workflow.get("payload")
                    if payload is None:
                        payload = {}
                    if not isinstance(payload, dict):
                        raise OrchestratorError(
                            "Workflow payload must be a dict",
                            details={
                                "index": index,
                                "workflow_id": workflow_id,
                                "payload_type": type(payload).__name__,
                            },
                        )

                    workflow_payload = {**workflow, "payload": {**normalized_input, **payload}}
                    result = run_workflow(workflow_payload)
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
                            "error": _error_dict(exc),
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

        evaluator_input = _resolve_evaluator_input(normalized_config, results)
        if evaluator_input is not None:
            try:
                evaluation_result = evaluate_roi(evaluator_input)
            except EvaluatorError as exc:
                error_count += 1
                errors.append(
                    {
                        "workflow_id": None,
                        "index": None,
                        "error": _error_dict(exc),
                    }
                )
            except Exception as exc:
                error_count += 1
                errors.append(
                    {
                        "workflow_id": None,
                        "index": None,
                        "error": _format_unexpected_error(exc),
                    }
                )
    except MenaceError as exc:
        errors.append({"workflow_id": None, "index": None, "error": _error_dict(exc)})
        error_count += 1
    except Exception as exc:
        logger.exception("Unexpected orchestrator failure")
        errors.append({"workflow_id": None, "index": None, "error": _format_unexpected_error(exc)})
        error_count += 1

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
        "data": {"results": results, "evaluation": evaluation_result},
        "errors": errors,
        "meta": {
            "workflow_count": len(workflows) if isinstance(workflows, list) else 0,
            "result_count": len(results),
            "ok_count": ok_count,
            "error_count": error_count,
        },
    }
