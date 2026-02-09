"""Failure guard utility for capturing critical stage failures."""

from __future__ import annotations

import json
import logging
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from shared_event_bus import event_bus as shared_event_bus

from failure_registry import (
    DEFAULT_REGISTRY_PATH,
    is_cooling_down,
    is_new_failure,
    record_failure,
)

LOGGER = logging.getLogger(__name__)

FAILURE_REGISTRY_PATH = Path("maintenance-logs/failure_registry.json")
FAILURE_AGGREGATE_REGISTRY_PATH = DEFAULT_REGISTRY_PATH


def _normalize_failure_payload(
    *,
    exc: BaseException,
    stage: str,
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "timestamp": time.time(),
        "stage": stage,
        "metadata": dict(metadata or {}),
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "stack": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }


def _load_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "failures": {}, "last_updated": time.time()}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        LOGGER.debug("failed to read failure registry", exc_info=True)
        return {"version": 1, "failures": {}, "last_updated": time.time()}
    if not isinstance(data, dict):
        return {"version": 1, "failures": {}, "last_updated": time.time()}
    data.setdefault("version", 1)
    data.setdefault("failures", {})
    return data


def _persist_failure_payload(
    payload: Mapping[str, Any],
    *,
    registry_path: Path,
    logger: logging.Logger | None,
) -> dict[str, Any]:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry = _load_registry(registry_path)
    failures = registry.get("failures")
    if not isinstance(failures, dict):
        failures = {}
    payload_id = str(payload.get("id") or uuid.uuid4())
    entry = {"id": payload_id, "payload": dict(payload)}
    failures[payload_id] = entry
    registry["failures"] = failures
    registry["last_updated"] = time.time()
    try:
        registry_path.write_text(
            json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8"
        )
    except Exception:
        if logger:
            logger.exception(
                "failed to persist failure registry",
                extra={"event": "failure-registry-write-error"},
            )
    return {
        "context_path": str(registry_path),
        "context_id": payload_id,
        "record": entry,
    }


def _publish_failure_event(
    payload: Mapping[str, Any],
    *,
    context: Mapping[str, Any],
    stage: str,
    logger: logging.Logger | None,
) -> None:
    event_payload = {
        "event": "failure_guard",
        "stage": stage,
        "context_path": context.get("context_path"),
        "context_id": context.get("context_id"),
        "payload": dict(payload),
    }
    if shared_event_bus is not None:
        try:
            shared_event_bus.publish("failure.guard", event_payload)
        except Exception:
            if logger:
                logger.debug("failed to publish failure guard event", exc_info=True)


def _invoke_self_debug(
    *,
    repo_root: str | Path,
    workflow_db: str | Path | None,
    context_path: str | None,
    context_id: str | None,
    logger: logging.Logger | None,
) -> None:
    if not context_path:
        return
    try:
        from menace_sandbox import menace_workflow_self_debug
    except Exception:
        if logger:
            logger.exception(
                "failure guard could not import menace_workflow_self_debug",
                extra={"event": "failure-guard-self-debug-import-error"},
            )
        return

    args = [
        "--repo-root",
        str(Path(repo_root).resolve()),
        "--source-menace-id",
        "failure_guard",
        "--metrics-source",
        "failure_guard",
        "--failure-context-path",
        str(context_path),
    ]
    if context_id:
        args.extend(["--failure-context-id", context_id])
    if workflow_db:
        args.extend(["--workflow-db", str(workflow_db)])
    try:
        menace_workflow_self_debug.main(args)
    except Exception:
        if logger:
            logger.exception(
                "failure guard self-debug run failed",
                extra={"event": "failure-guard-self-debug-error"},
            )


@dataclass
class FailureGuard:
    stage: str
    metadata: Mapping[str, Any] | None = None
    logger: logging.Logger | None = None
    registry_path: Path = FAILURE_REGISTRY_PATH
    aggregate_registry_path: Path = FAILURE_AGGREGATE_REGISTRY_PATH
    repo_root: str | Path = "."
    workflow_db: str | Path | None = None
    suppress: bool = False
    on_failure: Callable[[Mapping[str, Any], Mapping[str, Any]], None] | None = None

    def __enter__(self) -> "FailureGuard":
        return self

    def __exit__(self, exc_type, exc, _tb) -> bool:
        if exc is None:
            return False
        logger = self.logger or LOGGER
        payload = _normalize_failure_payload(
            exc=exc,
            stage=self.stage,
            metadata=self.metadata,
        )
        aggregate_record = record_failure(
            exception_type=payload.get("exception_type", type(exc).__name__),
            message=payload.get("message", str(exc)),
            stage=self.stage,
            context=payload.get("metadata", {}),
            registry_path=self.aggregate_registry_path,
        )
        cooling_down = is_cooling_down(aggregate_record)
        new_failure = is_new_failure(aggregate_record)
        context = _persist_failure_payload(
            payload, registry_path=self.registry_path, logger=logger
        )
        logger.exception(
            "failure guard captured exception",
            extra={
                "event": "failure-guard-captured",
                "stage": self.stage,
                "context_path": context.get("context_path"),
                "context_id": context.get("context_id"),
                "failure_fingerprint": aggregate_record.get("fingerprint"),
                "failure_occurrences": aggregate_record.get("occurrences"),
                "failure_cooling_down": cooling_down,
                "failure_new": new_failure,
            },
        )
        if not cooling_down or new_failure:
            _publish_failure_event(
                payload, context=context, stage=self.stage, logger=logger
            )
        _invoke_self_debug(
            repo_root=self.repo_root,
            workflow_db=self.workflow_db,
            context_path=context.get("context_path"),
            context_id=context.get("context_id"),
            logger=logger,
        )
        if self.on_failure:
            try:
                self.on_failure(payload, context)
            except Exception:
                logger.debug("failure guard on_failure hook failed", exc_info=True)
        return self.suppress
