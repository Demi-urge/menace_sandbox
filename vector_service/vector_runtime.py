"""Bootstrap orchestrator for vector service readiness gates.

This module coordinates the three bootstrap signals expected by the
``bootstrap_readiness`` watchdog: ``db_index_load``, ``retriever_hydration``, and
``vector_seeding``. Each stage is executed sequentially and emits a heartbeat
update so downstream components can observe readiness progress.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Mapping, MutableMapping

from menace_sandbox.bootstrap_readiness import CORE_COMPONENTS
from menace_sandbox.bootstrap_timeout_policy import (
    emit_bootstrap_heartbeat,
    read_bootstrap_heartbeat,
)

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("VECTOR_SERVICE_SKIP_DISCOVERY", "1")

from menace_sandbox.vector_service.lazy_bootstrap import warmup_vector_service
from menace_sandbox.vector_service.vector_store import get_default_vector_store
from menace_sandbox.vector_service.vectorizer import SharedVectorService
from menace_sandbox.vector_service import seed_vectors


logger = logging.getLogger(__name__)

_INIT_LOCK = threading.Lock()
_INITIALISED = threading.Event()


def _mark_component_state(component: str, status: str) -> None:
    """Update the bootstrap heartbeat with the latest component state."""

    now = time.time()
    heartbeat = read_bootstrap_heartbeat() or {}
    readiness = heartbeat.get("readiness") if isinstance(heartbeat, Mapping) else {}
    if readiness is None:
        readiness = {}

    components: MutableMapping[str, str] = {}
    component_readiness: MutableMapping[str, Mapping[str, object]] = {}

    if isinstance(readiness, Mapping):
        raw_components = readiness.get("components")
        if isinstance(raw_components, Mapping):
            components.update({str(k): str(v) for k, v in raw_components.items()})

        raw_component_readiness = readiness.get("component_readiness")
        if isinstance(raw_component_readiness, Mapping):
            for key, value in raw_component_readiness.items():
                state = value if isinstance(value, Mapping) else {"status": value}
                component_readiness[str(key)] = dict(state)

    components[component] = status
    component_readiness[component] = {"status": status, "ts": now}

    all_ready = all(components.get(name) == "ready" for name in CORE_COMPONENTS)
    readiness_payload: dict[str, object] = {
        "components": dict(components),
        "component_readiness": dict(component_readiness),
        "ready": all_ready,
        "online": bool(all_ready or readiness.get("online")),
    }

    enriched = dict(heartbeat)
    enriched["readiness"] = readiness_payload

    emit_bootstrap_heartbeat(enriched)


def _load_vector_indexes() -> None:
    """Load the configured vector store so indexes are available."""

    store = get_default_vector_store(lazy=False)
    if store is None:
        logger.warning("vector store unavailable; skipping db index load")
        return
    try:
        store.load()
        logger.info("vector store loaded for bootstrap", extra={"backend": store.__class__.__name__})
    except Exception:
        logger.exception("vector store failed to load during bootstrap")
        raise


def _hydrate_retriever() -> None:
    """Warm the retriever stack to ensure handlers and caches are live."""

    os.environ.setdefault("MENACE_VECTOR_HEAVY_BOOTSTRAP", "1")
    os.environ.setdefault("MENACE_BOOTSTRAP_BUDGET", "999999")
    warmup_vector_service(
        hydrate_handlers=True,
        run_vectorise=False,
        warmup_lite=False,
        force_heavy=True,
        logger=logger,
    )


def _seed_vectors() -> None:
    """Seed baseline vectors to avoid cold-start requests."""

    service = SharedVectorService(
        hydrate_handlers=True,
        warmup_lite=False,
        lazy_vector_store=False,
    )
    seed_vectors.seed_initial_vectors(service, logger=logger)


def initialize_vector_service() -> Mapping[str, object] | None:
    """Run all vector bootstrap stages and emit readiness signals.

    Returns the latest heartbeat snapshot when available so callers can
    validate readiness gates before continuing with dependent initialisers.
    """

    if _INITIALISED.is_set():
        return read_bootstrap_heartbeat()
    with _INIT_LOCK:
        if _INITIALISED.is_set():
            return read_bootstrap_heartbeat()

        stages = (
            ("db_index_load", _load_vector_indexes),
            ("retriever_hydration", _hydrate_retriever),
            ("vector_seeding", _seed_vectors),
        )

        heartbeat_snapshot: Mapping[str, object] | None = None
        for component, action in stages:
            logger.info("starting vector bootstrap stage", extra={"component": component})
            try:
                action()
            except Exception as exc:
                heartbeat_snapshot = read_bootstrap_heartbeat()
                logger.exception(
                    "vector bootstrap stage failed",
                    extra={"component": component, "heartbeat": heartbeat_snapshot},
                )
                try:
                    _mark_component_state(component, "failed")
                finally:
                    pass
                raise VectorBootstrapError(component, heartbeat_snapshot, exc) from exc

            _mark_component_state(component, "ready")
            heartbeat_snapshot = read_bootstrap_heartbeat()
            logger.info(
                "completed vector bootstrap stage",
                extra={"component": component, "heartbeat": heartbeat_snapshot},
            )

        _INITIALISED.set()
        return heartbeat_snapshot


def main() -> None:
    """CLI entrypoint used by subprocesses or module execution."""

    logging.basicConfig(level=logging.INFO)
    initialize_vector_service()


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()

class VectorBootstrapError(RuntimeError):
    """Raised when a vector bootstrap stage fails."""

    def __init__(self, stage: str, heartbeat: Mapping[str, object] | None, exc: BaseException):
        self.stage = stage
        self.heartbeat = heartbeat
        super().__init__(f"vector bootstrap stage '{stage}' failed")
        self.__cause__ = exc

