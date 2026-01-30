"""Synchronous workflow orchestrator."""

from __future__ import annotations

import logging
from typing import Any

from menace.core.workflow_runner import run_workflow
from menace.errors.exceptions import (
    ConfigError,
    MenaceError,
    OrchestratorError,
    WorkflowExecutionError,
    WorkflowValidationError,
)
from menace.infra.config_loader import load_config as load_infra_config

logger = logging.getLogger(__name__)


def _error_payload(error: MenaceError) -> dict[str, Any]:
    """Convert a MenaceError into a structured error payload.

    Args:
        error (MenaceError): Domain-specific exception instance.

    Returns:
        dict[str, Any]: Structured error dictionary from ``error.to_dict()``.

    Raises:
        None: This helper does not raise.

    Invariants:
        - Output is deterministic for a given error instance.
        - No ``None`` values are introduced by this helper.
    """
    return error.to_dict()


def _unexpected_error_payload(exc: Exception) -> dict[str, Any]:
    """Create an orchestrator error payload for unexpected exceptions.

    Args:
        exc (Exception): Unexpected exception to serialize.

    Returns:
        dict[str, Any]: Structured ``OrchestratorError`` payload.

    Raises:
        None: This helper does not raise.

    Invariants:
        - Payload includes the exception type and message.
        - Output keys are stable for identical inputs.
    """
    return OrchestratorError(
        message="Unexpected orchestrator exception",
        details={
            "exception_type": type(exc).__name__,
            "message": str(exc),
        },
    ).to_dict()


def _validate_workflow_shape(workflow: Any, index: int) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Validate the basic schema of a workflow definition.

    Args:
        workflow (Any): Candidate workflow payload to validate.
        index (int): Index of the workflow in the batch for error context.

    Returns:
        tuple[dict[str, Any] | None, dict[str, Any] | None]: Tuple of
        (validated workflow dict or ``None``, error payload or ``None``).

    Raises:
        None: Validation errors are returned as payloads, not raised.

    Invariants:
        - ``workflow`` must be a dict with required keys and non-empty
          ``workflow_id``.
        - No mutation of the input ``workflow`` occurs.
        - Validation is deterministic for identical inputs.
    """
    if workflow is None:
        error = WorkflowValidationError(
            message="Workflow definition cannot be None.",
            details={"index": index},
        )
        return None, _error_payload(error)

    if not isinstance(workflow, dict):
        error = WorkflowValidationError(
            message="Workflow definition must be a dictionary.",
            details={"index": index, "received_type": type(workflow).__name__},
        )
        return None, _error_payload(error)

    missing_keys = [key for key in ("workflow_id", "steps", "payload") if key not in workflow]
    if missing_keys:
        error = WorkflowValidationError(
            message="Workflow definition missing required keys.",
            details={"index": index, "missing_keys": missing_keys},
        )
        return None, _error_payload(error)

    workflow_id = workflow.get("workflow_id")
    if not isinstance(workflow_id, str) or not workflow_id.strip():
        error = WorkflowValidationError(
            message="workflow_id must be a non-empty string.",
            details={"index": index, "workflow_id": workflow_id},
        )
        return None, _error_payload(error)

    steps = workflow.get("steps")
    if not isinstance(steps, list):
        error = WorkflowValidationError(
            message="steps must be a list of step definitions.",
            details={"index": index, "workflow_id": workflow_id, "steps_type": type(steps).__name__},
        )
        return None, _error_payload(error)

    payload = workflow.get("payload")
    if not isinstance(payload, dict):
        error = WorkflowValidationError(
            message="payload must be a dictionary.",
            details={"index": index, "workflow_id": workflow_id, "payload_type": type(payload).__name__},
        )
        return None, _error_payload(error)

    metadata = workflow.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        error = WorkflowValidationError(
            message="metadata must be a dictionary when provided.",
            details={"index": index, "workflow_id": workflow_id, "metadata_type": type(metadata).__name__},
        )
        return None, _error_payload(error)

    return workflow, None


def run_orchestrator(workflows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Run workflows synchronously and return deterministic aggregated results.

    Contract:
        - ``workflows`` must be a list (empty list is allowed and returns zeroed
          results).
        - Each workflow must be a dictionary containing ``workflow_id`` (str),
          ``steps`` (list), ``payload`` (dict), and optional ``metadata`` (dict).
        - ``config`` is validated through :func:`menace.infra.config_loader.load_config`.
        - Workflows run in input order exactly once (no recursion, retries, or
          reordering).
        - This function never raises; all failures are returned in ``errors``.

    Returns:
        A dictionary with the schema:
        - ``status``: ``ok`` | ``error``
        - ``data``: ``{"results": [...], "config": {...}}``
        - ``errors``: list of deterministic error payloads
        - ``metadata``: counts + status summary + config validation metadata

        Each entry in ``data["results"]`` is a workflow result that includes a
        top-level ``metadata`` key. The workflow ``metadata`` payload includes
        ``workflow_id``, any input ``metadata`` fields, and deterministic failure
        indicators: ``partial_failure`` (bool), ``error_count`` (int), and
        ``failed_steps`` (list[int]).
    """

    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    ok_count = 0
    error_count = 0
    config_metadata: dict[str, Any] | None = None
    normalized_config: dict[str, Any] | None = None

    try:
        if workflows is None:
            error = WorkflowValidationError(
                message="workflows cannot be None.",
                details={"received": None},
            )
            return _final_response(
                status="error",
                workflows_count=0,
                results=results,
                errors=[{"workflow_id": None, "index": None, "error": _error_payload(error)}],
                ok_count=0,
                error_count=1,
                config_meta=None,
                normalized_config=None,
            )

        if not isinstance(workflows, list):
            error = WorkflowValidationError(
                message="workflows must be a list.",
                details={"received_type": type(workflows).__name__},
            )
            return _final_response(
                status="error",
                workflows_count=0,
                results=results,
                errors=[{"workflow_id": None, "index": None, "error": _error_payload(error)}],
                ok_count=0,
                error_count=1,
                config_meta=None,
                normalized_config=None,
            )

        try:
            config_response = load_infra_config(config)
            normalized_config = config_response["data"]
            config_metadata = config_response.get("metadata")
        except ConfigError as exc:
            errors.append({"workflow_id": None, "index": None, "error": _error_payload(exc)})
            return _final_response(
                status="error",
                workflows_count=len(workflows),
                results=results,
                errors=errors,
                ok_count=0,
                error_count=1,
                config_meta=None,
                normalized_config=None,
            )

        if len(workflows) == 0:
            logger.info("No workflows provided; returning empty results")
            return _final_response(
                status="ok",
                workflows_count=0,
                results=results,
                errors=errors,
                ok_count=0,
                error_count=0,
                config_meta=config_metadata,
                normalized_config=normalized_config,
            )

        for index, workflow in enumerate(workflows):
            workflow_id: str | None = None
            validated_workflow, shape_error = _validate_workflow_shape(workflow, index)
            if shape_error is not None:
                error_count += 1
                errors.append({"workflow_id": workflow_id, "index": index, "error": shape_error})
                continue

            assert validated_workflow is not None
            workflow_id = validated_workflow.get("workflow_id")
            try:
                result = run_workflow(validated_workflow)
                results.append(result)
                workflow_status = result.get("status")
                workflow_errors = result.get("errors", [])
                workflow_metadata = result.get("metadata", {})
                if workflow_status == "ok":
                    ok_count += 1
                else:
                    error_count += 1
                    error = WorkflowExecutionError(
                        message="Workflow completed with errors.",
                        details={
                            "workflow_id": workflow_id,
                            "index": index,
                            "status": workflow_status,
                            "errors": workflow_errors,
                            "metadata": workflow_metadata,
                        },
                    )
                    errors.append(
                        {
                            "workflow_id": workflow_id,
                            "index": index,
                            "error": _error_payload(error),
                        }
                    )
            except MenaceError as exc:
                error_count += 1
                errors.append(
                    {
                        "workflow_id": workflow_id,
                        "index": index,
                        "error": _error_payload(exc),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - unexpected workflow errors
                error_count += 1
                errors.append(
                    {
                        "workflow_id": workflow_id,
                        "index": index,
                        "error": _unexpected_error_payload(exc),
                    }
                )

        status = _derive_status(ok_count, error_count)
        return _final_response(
            status=status,
            workflows_count=len(workflows),
            results=results,
            errors=errors,
            ok_count=ok_count,
            error_count=error_count,
            config_meta=config_metadata,
            normalized_config=normalized_config,
        )
    except Exception as exc:  # noqa: BLE001 - top-level safety net
        logger.exception("Unexpected orchestrator failure")
        return _final_response(
            status="error",
            workflows_count=len(workflows) if isinstance(workflows, list) else 0,
            results=results,
            errors=[{"workflow_id": None, "index": None, "error": _unexpected_error_payload(exc)}],
            ok_count=0,
            error_count=1,
            config_meta=None,
            normalized_config=None,
        )


def _derive_status(ok_count: int, error_count: int) -> str:
    """Return aggregate status based on success/failure counts.

    Args:
        ok_count (int): Count of successful workflows.
        error_count (int): Count of workflows with errors.

    Returns:
        str: ``"ok"`` when at least one success and no errors; otherwise
        ``"error"``.

    Raises:
        None: This helper does not raise.

    Invariants:
        - Deterministic mapping for given counts.
        - Any non-zero error count yields ``"error"``.
    """
    if error_count:
        return "error"
    if ok_count:
        return "ok"
    return "error"


def _final_response(
    *,
    status: str,
    workflows_count: int,
    results: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    ok_count: int,
    error_count: int,
    config_meta: dict[str, Any] | None,
    normalized_config: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the deterministic orchestrator response payload.

    Args:
        status (str): Aggregated status string.
        workflows_count (int): Total number of workflows processed.
        results (list[dict[str, Any]]): Collected workflow results.
        errors (list[dict[str, Any]]): Collected structured errors.
        ok_count (int): Number of workflows with ``"ok"`` status.
        error_count (int): Number of workflows with ``"error"`` status.
        config_meta (dict[str, Any] | None): Optional config validation metadata.
        normalized_config (dict[str, Any] | None): Optional normalized config data.

    Returns:
        dict[str, Any]: Final response payload with data, errors, and metadata.

    Raises:
        None: This helper does not raise.

    Invariants:
        - Output keys are stable for identical inputs.
        - ``partial_failure`` is true only when both ok and error counts exist.
    """
    metadata = {
        "workflow_count": workflows_count,
        "result_count": len(results),
        "ok_count": ok_count,
        "error_count": error_count,
        "partial_failure": bool(ok_count and error_count),
        "status_summary": {
            "ok": ok_count,
            "error": error_count,
        },
        "config_metadata": config_meta,
    }
    return {
        "status": status,
        "data": {
            "results": results,
            "config": normalized_config,
        },
        "errors": errors,
        "metadata": metadata,
    }
