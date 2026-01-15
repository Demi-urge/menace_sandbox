"""Helpers for seeding baseline vectors during bootstrap."""

from __future__ import annotations

import logging
import os
import time
import urllib.request
import uuid
from typing import Iterable

from .vector_store import VectorStore, get_default_vector_store

try:  # pragma: no cover - optional dependency
    from .vectorizer import SharedVectorService
except Exception:  # pragma: no cover - fallback when vectorizer import fails

    class SharedVectorService:  # type: ignore
        def __init__(self, *_, **__):
            raise RuntimeError("SharedVectorService unavailable")


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


def _wait_for_vector_service_ready(*, logger: logging.Logger) -> bool:
    base = os.environ.get("VECTOR_SERVICE_URL")
    if not base:
        return True

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
                return True
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
                return True
        except Exception as exc:
            logger.debug("vector service readiness probe failed: %s", exc)
        time.sleep(delay)
        delay = min(delay * 1.5, 5.0)

    elapsed = time.monotonic() - start
    logger.warning(
        "vector service readiness timed out after extended retries; falling back to local seeding",
        extra={"retry_budget": retry_budget, "elapsed": elapsed, "attempts": attempts},
    )
    return False


def seed_initial_vectors(service: SharedVectorService | None, *, logger: logging.Logger) -> None:
    """Warm baseline embeddings for the configured vector store."""

    store = get_default_vector_store(lazy=False)
    if store is None:
        logger.warning("vector store unavailable; skipping vector seeding")
        return

    try:
        store.load()
    except Exception:  # pragma: no cover - defensive; seeding still proceeds
        logger.debug("vector store load failed during seeding", exc_info=True)

    if service is None:
        _seed_direct(store, logger=logger)
        return

    if not _wait_for_vector_service_ready(logger=logger):
        _seed_direct(store, logger=logger)
        return

    try:
        text = "menace bootstrap seed vector"
        for _ in range(3):
            record_id = uuid.uuid4().hex
            service.vectorise_and_store(
                "text", record_id, {"text": text}, metadata={"seed": True}
            )
        logger.info("seeded baseline vectors via SharedVectorService", extra={"count": 3})
    except Exception:
        logger.exception("shared vector service seeding failed; falling back to direct store")
        _seed_direct(store, logger=logger)


__all__ = ["seed_initial_vectors"]
