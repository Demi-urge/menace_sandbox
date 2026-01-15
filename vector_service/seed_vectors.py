"""Helpers for seeding baseline vectors during bootstrap."""

from __future__ import annotations

import logging
import os
import time
import urllib.error
import urllib.request
import uuid
from typing import Iterable, Literal, Mapping, MutableMapping, TypedDict

from .vector_store import VectorStore, get_default_vector_store

try:  # pragma: no cover - optional dependency
    from .vectorizer import SharedVectorService
except Exception:  # pragma: no cover - fallback when vectorizer import fails

    class SharedVectorService:  # type: ignore
        def __init__(self, *_, **__):
            raise RuntimeError("SharedVectorService unavailable")


def _emit_local_seed_readiness(*, elapsed: float, logger: logging.Logger) -> None:
    try:  # pragma: no cover - best effort
        from menace_sandbox.bootstrap_timeout_policy import (
            emit_bootstrap_heartbeat,
            read_bootstrap_heartbeat,
        )
    except Exception:
        return

    heartbeat = read_bootstrap_heartbeat() or {}
    readiness = heartbeat.get("readiness") if isinstance(heartbeat, Mapping) else {}

    components: MutableMapping[str, str] = {}
    component_readiness: MutableMapping[str, Mapping[str, object]] = {}

    if isinstance(readiness, Mapping):
        raw_components = readiness.get("components")
        if isinstance(raw_components, Mapping):
            components.update({str(key): str(value) for key, value in raw_components.items()})

        raw_component_readiness = readiness.get("component_readiness")
        if isinstance(raw_component_readiness, Mapping):
            for key, value in raw_component_readiness.items():
                state = value if isinstance(value, Mapping) else {"status": value}
                component_readiness[str(key)] = dict(state)

    now = time.time()
    components["vector_seeding"] = "ready"
    readiness_entry: MutableMapping[str, object] = dict(
        component_readiness.get("vector_seeding", {})
    )
    readiness_entry.update(
        {
            "status": "ready",
            "ts": now,
            "reason": "fallback_local_seed",
            "elapsed": elapsed,
        }
    )
    component_readiness["vector_seeding"] = dict(readiness_entry)

    readiness_payload: dict[str, object] = {
        "components": dict(components),
        "component_readiness": dict(component_readiness),
        "ready": readiness.get("ready") if isinstance(readiness, Mapping) else False,
        "online": readiness.get("online") if isinstance(readiness, Mapping) else False,
    }

    enriched = dict(heartbeat)
    enriched["readiness"] = readiness_payload
    emit_bootstrap_heartbeat(enriched)
    logger.debug(
        "emitted local vector seeding readiness heartbeat",
        extra={"reason": "fallback_local_seed", "elapsed": elapsed},
    )


def _seed_direct(store: VectorStore, *, logger: logging.Logger) -> None:
    """Insert a handful of placeholder vectors into the store."""

    # Best-effort seeding so downstream retrievals never start from an empty
    # index. When the store exposes a ``dim`` attribute we generate a matching
    # zero vector; otherwise fall back to a tiny vector that most backends
    # accept.
    dimension = getattr(store, "dim", 3) or 3
    vector = [0.0] * int(dimension)
    seed_ids: Iterable[str] = (f"bootstrap-{uuid.uuid4().hex}" for _ in range(3))
    for record_id in seed_ids:
        try:
            store.add("bootstrap", record_id, vector, metadata={"seed": True})
        except Exception:  # pragma: no cover - seed best-effort
            logger.debug("vector store seed failed for %s", record_id, exc_info=True)
    logger.info("vector store seeded", extra={"count": 3, "dimension": dimension})


class VectorSeedStatus(TypedDict):
    status: Literal["remote", "local", "skipped"]
    elapsed: float
    attempts: int


class VectorServiceReadyStatus(TypedDict):
    ready: bool
    elapsed: float
    attempts: int
    timed_out: bool


def _wait_for_vector_service_ready(*, logger: logging.Logger) -> VectorServiceReadyStatus:
    base = os.environ.get("VECTOR_SERVICE_URL")
    if not base:
        return {"ready": True, "elapsed": 0.0, "attempts": 0, "timed_out": False}

    try:
        timeout = max(1.0, float(os.environ.get("VECTOR_SERVICE_READY_TIMEOUT", "150")))
    except Exception:
        logger.warning(
            "invalid VECTOR_SERVICE_READY_TIMEOUT=%r; defaulting to 150s",
            os.environ.get("VECTOR_SERVICE_READY_TIMEOUT"),
        )
        timeout = 150.0

    try:
        retry_budget = max(0, int(os.environ.get("VECTOR_SERVICE_READY_RETRY_BUDGET", "6")))
    except Exception:
        logger.warning(
            "invalid VECTOR_SERVICE_READY_RETRY_BUDGET=%r; defaulting to 6 retries",
            os.environ.get("VECTOR_SERVICE_READY_RETRY_BUDGET"),
        )
        retry_budget = 6

    ready_url = f"{base.rstrip('/')}/health/ready"
    start = time.monotonic()
    deadline = start + timeout
    delay = 1.0
    attempts = 0
    while time.monotonic() < deadline:
        try:
            attempts += 1
            with urllib.request.urlopen(ready_url, timeout=2.0):
                return {
                    "ready": True,
                    "elapsed": time.monotonic() - start,
                    "attempts": attempts,
                    "timed_out": False,
                }
        except Exception as exc:
            logger.debug("vector service readiness probe failed: %s", exc)
        time.sleep(delay)
        delay = min(delay * 1.5, 5.0)

    for _ in range(retry_budget):
        try:
            attempts += 1
            with urllib.request.urlopen(ready_url, timeout=2.0):
                elapsed = time.monotonic() - start
                logger.info(
                    "vector service readiness reached during extended retries",
                    extra={"retry_budget": retry_budget, "elapsed": elapsed, "attempts": attempts},
                )
                return {
                    "ready": True,
                    "elapsed": elapsed,
                    "attempts": attempts,
                    "timed_out": False,
                }
        except Exception as exc:
            logger.debug("vector service readiness probe failed: %s", exc)
        time.sleep(delay)
        delay = min(delay * 1.5, 5.0)

    elapsed = time.monotonic() - start
    logger.warning(
        "vector service readiness timed out after extended retries; falling back to local seeding",
        extra={"retry_budget": retry_budget, "elapsed": elapsed, "attempts": attempts},
    )
    return {
        "ready": False,
        "elapsed": elapsed,
        "attempts": attempts,
        "timed_out": True,
    }


def seed_initial_vectors(
    service: SharedVectorService | None, *, logger: logging.Logger
) -> VectorSeedStatus:
    """Warm baseline embeddings for the configured vector store."""

    seed_start = time.monotonic()
    store = get_default_vector_store(lazy=False)
    if store is None:
        logger.warning("vector store unavailable; skipping vector seeding")
        return {"status": "skipped", "elapsed": 0.0, "attempts": 0}

    try:
        store.load()
    except Exception:  # pragma: no cover - defensive; seeding still proceeds
        logger.debug("vector store load failed during seeding", exc_info=True)

    if service is None:
        _seed_direct(store, logger=logger)
        elapsed = time.monotonic() - seed_start
        _emit_local_seed_readiness(elapsed=elapsed, logger=logger)
        logger.info(
            "seeding recovered via local fallback",
            extra={"elapsed": elapsed},
        )
        return {"status": "local", "elapsed": 0.0, "attempts": 0}

    ready_status = _wait_for_vector_service_ready(logger=logger)
    if not ready_status["ready"]:
        _seed_direct(store, logger=logger)
        elapsed = time.monotonic() - seed_start
        _emit_local_seed_readiness(elapsed=elapsed, logger=logger)
        logger.info(
            "seeding recovered via local fallback",
            extra={"elapsed": elapsed},
        )
        if ready_status["timed_out"]:
            logger.info(
                "local seeding completed after remote readiness timeout",
                extra={
                    "elapsed": ready_status["elapsed"],
                    "attempts": ready_status["attempts"],
                },
            )
        return {
            "status": "local",
            "elapsed": ready_status["elapsed"],
            "attempts": ready_status["attempts"],
        }

    def _is_expected_deferral(exc: Exception) -> bool:
        if isinstance(exc, (TimeoutError, urllib.error.URLError)):
            return True
        if isinstance(exc, RuntimeError):
            message = str(exc).lower()
            return "deferred" in message or "unavailable" in message
        return False

    try:
        text = "menace bootstrap seed vector"
        for _ in range(3):
            record_id = uuid.uuid4().hex
            service.vectorise_and_store(
                "text", record_id, {"text": text}, metadata={"seed": True}
            )
        logger.info("seeded baseline vectors via SharedVectorService", extra={"count": 3})
        return {
            "status": "remote",
            "elapsed": ready_status["elapsed"],
            "attempts": ready_status["attempts"],
        }
    except Exception as exc:
        if _is_expected_deferral(exc):
            logger.warning(
                "shared vector service unavailable; falling back to direct store",
                extra={"error": str(exc)},
            )
        else:
            logger.exception(
                "shared vector service seeding failed; falling back to direct store"
            )
        _seed_direct(store, logger=logger)
        elapsed = time.monotonic() - seed_start
        _emit_local_seed_readiness(elapsed=elapsed, logger=logger)
        logger.info(
            "vector store seeded via fallback after remote/local deferral",
            extra={"elapsed": elapsed},
        )
        logger.info(
            "seeding recovered via local fallback",
            extra={"elapsed": elapsed},
        )
        return {"status": "local", "elapsed": 0.0, "attempts": 0}


__all__ = ["seed_initial_vectors"]
