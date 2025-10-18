from __future__ import annotations

"""Simple wrapper combining :mod:`gpt_memory` and :mod:`gpt_knowledge_service`.

This module exposes :class:`LocalKnowledgeModule` which couples
:class:`gpt_memory.GPTMemoryManager` with :class:`gpt_knowledge_service.GPTKnowledgeService`.
It provides a minimal interface used by bots that need persistent long-term
context.
"""

import importlib.util
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Sequence

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - keep import lightweight
    SentenceTransformer = None  # type: ignore

_HELPER_NAME = "import_compat"
_PACKAGE_NAME = "menace_sandbox"

try:  # pragma: no cover - prefer package import when installed
    from menace_sandbox import import_compat  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - support flat execution
    _helper_path = Path(__file__).resolve().parent / f"{_HELPER_NAME}.py"
    _spec = importlib.util.spec_from_file_location(
        f"{_PACKAGE_NAME}.{_HELPER_NAME}",
        _helper_path,
    )
    if _spec is None or _spec.loader is None:  # pragma: no cover - defensive
        raise
    import_compat = importlib.util.module_from_spec(_spec)
    sys.modules[f"{_PACKAGE_NAME}.{_HELPER_NAME}"] = import_compat
    sys.modules[_HELPER_NAME] = import_compat
    _spec.loader.exec_module(import_compat)
else:  # pragma: no cover - ensure helper aliases exist
    sys.modules.setdefault(_HELPER_NAME, import_compat)
    sys.modules.setdefault(f"{_PACKAGE_NAME}.{_HELPER_NAME}", import_compat)

import_compat.bootstrap(__name__, __file__)
load_internal = import_compat.load_internal

get_embedder = load_internal("governed_embeddings").get_embedder

GPTMemoryManager = load_internal("gpt_memory").GPTMemoryManager
GPTKnowledgeService = load_internal("gpt_knowledge_service").GPTKnowledgeService
vector_service_module = load_internal("vector_service")
CognitionLayer = vector_service_module.CognitionLayer
SharedVectorService = getattr(vector_service_module, "SharedVectorService", None)

try:  # pragma: no cover - canonical tag constants
    _log_tags = load_internal("log_tags")
except ModuleNotFoundError:  # pragma: no cover - fallback when tags unavailable
    FEEDBACK = "feedback"
    IMPROVEMENT_PATH = "improvement_path"
    ERROR_FIX = "error_fix"
except Exception:  # pragma: no cover - degrade gracefully
    FEEDBACK = "feedback"
    IMPROVEMENT_PATH = "improvement_path"
    ERROR_FIX = "error_fix"
else:
    FEEDBACK = _log_tags.FEEDBACK
    IMPROVEMENT_PATH = _log_tags.IMPROVEMENT_PATH
    ERROR_FIX = _log_tags.ERROR_FIX


logger = logging.getLogger(__name__)

try:
    _EMBEDDER_TIMEOUT = float(os.getenv("LOCAL_KNOWLEDGE_EMBEDDER_TIMEOUT", "10"))
except Exception:
    _EMBEDDER_TIMEOUT = 10.0
    logger.warning("invalid LOCAL_KNOWLEDGE_EMBEDDER_TIMEOUT; defaulting to 10s")
else:
    if _EMBEDDER_TIMEOUT < 0:
        logger.warning(
            "LOCAL_KNOWLEDGE_EMBEDDER_TIMEOUT must be non-negative; defaulting to 10s"
        )
        _EMBEDDER_TIMEOUT = 10.0


class LocalKnowledgeModule:
    """Aggregate GPT memory and summarised insights.

    Parameters
    ----------
    db_path:
        Location of the SQLite database used for storing interactions.  If an
        existing :class:`GPTMemoryManager` is supplied it will be used instead.
    manager:
        Optional pre-initialised :class:`GPTMemoryManager` instance.
    service:
        Optional pre-initialised :class:`GPTKnowledgeService` instance.  When not
        provided one will be created using ``manager``.
    embedder:
        Optional :class:`SentenceTransformer` shared with the underlying
        :class:`GPTMemoryManager` to enable semantic search.
    """

    def __init__(
        self,
        db_path: str | Path = "gpt_memory.db",
        *,
        manager: GPTMemoryManager | None = None,
        service: GPTKnowledgeService | None = None,
        embedder: "SentenceTransformer | None" = None,
        cognition_layer: CognitionLayer | None = None,
    ) -> None:
        self.memory = manager or GPTMemoryManager(db_path, embedder=embedder)
        self.knowledge = service or GPTKnowledgeService(self.memory)
        self.cognition_layer = cognition_layer

    # ------------------------------------------------------------------ facade
    def log(
        self, prompt: str, response: str, tags: Sequence[str] | None = None
    ) -> None:
        """Store an interaction in long-term memory."""

        self.memory.log_interaction(prompt, response, tags)

    def get_insights(self, tag: str) -> str:
        """Return the latest generated insight for ``tag``."""

        return self.knowledge.get_recent_insights(tag)

    def refresh(self) -> None:
        """Regenerate stored insights from recent interactions."""

        self.knowledge.update_insights()

    # ---------------------------------------------------------------- context
    def _fmt(self, entries: Sequence[Any], title: str) -> str:
        parts: list[str] = []
        for e in entries:
            prompt = getattr(e, "prompt", "")
            response = getattr(e, "response", "")
            text = response or prompt
            if text:
                parts.append(f"- {text}")
        if not parts:
            return ""
        body = "\n".join(parts)
        return f"### {title}\n{body}"

    def build_context(self, key: str, limit: int = 5) -> str:
        """Return formatted context for ``key`` using :class:`CognitionLayer`.

        Falls back to the legacy memory aggregation when no
        :class:`CognitionLayer` instance is available.
        """

        if self.cognition_layer is not None:
            try:
                ctx, _ = self.cognition_layer.query(key, top_k=limit)
                return ctx
            except Exception:
                return ""

        sections: list[str] = []
        try:
            fb = self.memory.search_context(
                key, tags=[FEEDBACK], limit=limit, use_embeddings=True
            )
            ctx = self._fmt(fb, "Feedback")
            if ctx:
                sections.append(ctx)
        except Exception:
            pass
        try:
            fixes = self.memory.search_context(
                key, tags=[ERROR_FIX], limit=limit, use_embeddings=True
            )
            ctx = self._fmt(fixes, "Error fixes")
            if ctx:
                sections.append(ctx)
        except Exception:
            pass
        try:
            improv = self.memory.search_context(
                key, tags=[IMPROVEMENT_PATH], limit=limit, use_embeddings=True
            )
            ctx = self._fmt(improv, "Improvement paths")
            if ctx:
                sections.append(ctx)
        except Exception:
            pass

        try:
            insight = self.knowledge.get_recent_insights(FEEDBACK)
            if insight:
                sections.append(f"### Feedback insight\n- {insight}")
        except Exception:
            pass
        try:
            insight = self.knowledge.get_recent_insights(ERROR_FIX)
            if insight:
                sections.append(f"### Error fix insight\n- {insight}")
        except Exception:
            pass
        try:
            insight = self.knowledge.get_recent_insights(IMPROVEMENT_PATH)
            if insight:
                sections.append(f"### Improvement path insight\n- {insight}")
        except Exception:
            pass

        return "\n\n".join(sections)


_LOCAL_KNOWLEDGE: "LocalKnowledgeModule | None" = None


def _attach_embedder(module: "LocalKnowledgeModule", embedder: "SentenceTransformer") -> None:
    """Wire ``embedder`` into ``module`` and its memory backend."""

    try:
        module.memory.embedder = embedder
    except Exception:
        logger.exception("failed to attach embedder to memory manager")
        return

    try:
        vector_service = getattr(module.memory, "vector_service", None)
        if vector_service is None and SharedVectorService is not None:
            module.memory.vector_service = SharedVectorService(embedder)
        elif vector_service is not None and hasattr(vector_service, "text_embedder"):
            vector_service.text_embedder = embedder
    except Exception:
        logger.exception("failed to attach embedder to vector service")
        return

    logger.info(
        "sentence transformer attached to LocalKnowledgeModule",
        extra={"embedder": type(embedder).__name__},
    )


def _load_embedder_async(
    wait_time: float | None,
) -> tuple[dict[str, "SentenceTransformer | None"], threading.Thread, list[BaseException]]:
    """Launch ``get_embedder`` in a background thread and wait up to ``wait_time``."""

    result: dict[str, "SentenceTransformer | None"] = {"embedder": None}
    errors: list[BaseException] = []

    def _worker() -> None:
        try:
            embedder = get_embedder(timeout=None)
        except BaseException as exc:  # pragma: no cover - diagnostics only
            errors.append(exc)
            return
        result["embedder"] = embedder

    thread = threading.Thread(
        target=_worker,
        name="local-knowledge-embedder",
        daemon=True,
    )
    thread.start()

    join_timeout = None
    if wait_time is not None:
        join_timeout = max(0.0, wait_time)
    thread.join(join_timeout)

    return result, thread, errors


def init_local_knowledge(mem_db: str | Path) -> LocalKnowledgeModule:
    """Return a process-wide :class:`LocalKnowledgeModule` instance.

    The module is initialised on first use and subsequent calls return the
    existing instance, ensuring that all components share the same underlying
    :class:`GPTMemoryManager` and :class:`GPTKnowledgeService`.
    """

    global _LOCAL_KNOWLEDGE
    if _LOCAL_KNOWLEDGE is None:
        start = time.perf_counter()
        logger.info(
            "initialising LocalKnowledgeModule",
            extra={"memory_db": str(mem_db)},
        )
        embedder_state, thread, errors = _load_embedder_async(_EMBEDDER_TIMEOUT)
        embedder = embedder_state.get("embedder")
        if thread.is_alive() and _EMBEDDER_TIMEOUT:
            logger.warning(
                "sentence transformer still initialising after %.1fs; continuing without embeddings",
                _EMBEDDER_TIMEOUT,
            )
        if errors:
            logger.warning(
                "sentence transformer initialisation raised: %s", errors[-1]
            )

        module = LocalKnowledgeModule(mem_db, embedder=embedder)

        def _finalise_embedder(
            state: dict[str, "SentenceTransformer | None"],
            err_state: list[BaseException],
        ) -> None:
            thread.join()
            if err_state:
                logger.warning(
                    "sentence transformer initialisation raised: %s", err_state[-1]
                )
            ready = state.get("embedder")
            if ready is None:
                ready = get_embedder(timeout=0)
            if ready is None:
                return
            _attach_embedder(module, ready)

        if thread.is_alive():
            threading.Thread(
                target=_finalise_embedder,
                args=(embedder_state, errors),
                name="local-knowledge-embedder-finalise",
                daemon=True,
            ).start()
        else:
            ready = embedder_state.get("embedder")
            if ready is not None:
                _attach_embedder(module, ready)

        _LOCAL_KNOWLEDGE = module
        duration = time.perf_counter() - start
        logger.info(
            "LocalKnowledgeModule ready",
            extra={
                "memory_db": str(mem_db),
                "duration": round(duration, 3),
                "embedder_available": embedder is not None,
                "embedder_thread_alive": thread.is_alive(),
            },
        )
    return _LOCAL_KNOWLEDGE


__all__ = ["LocalKnowledgeModule", "init_local_knowledge"]
