from __future__ import annotations

"""Runtime registry for vectoriser and database plugins.

This module allows new record modalities to be registered without modifying
code.  Each registration maps a short ``kind`` to the module and class
implementing the vectoriser.  Optional database information can also be
provided so :mod:`embedding_backfill` discovers the appropriate
``EmbeddableDBMixin`` implementation automatically.
"""

from typing import Callable, Dict, Tuple, Optional
import importlib
import pkgutil
import inspect
import logging
import os
import time


logger = logging.getLogger(__name__)

# Registry mapping kind -> (vectoriser module, vectoriser class,
#                           db module, db class)
_VECTOR_REGISTRY: Dict[str, Tuple[str, str, Optional[str], Optional[str]]] = {}


def register_vectorizer(
    kind: str,
    module_path: str,
    class_name: str,
    *,
    db_module: str | None = None,
    db_class: str | None = None,
) -> None:
    """Register a vectoriser and optional database implementation."""

    _VECTOR_REGISTRY[kind.lower()] = (module_path, class_name, db_module, db_class)


def _accepts_bootstrap_fast(target: type[object]) -> bool:
    try:
        parameters = inspect.signature(target).parameters
    except (TypeError, ValueError):
        return False

    return "bootstrap_fast" in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def _patch_stub_handler(record: Dict[str, any]) -> list[float]:
    return []


# Mark stub so callers can detect deferred bootstrap handling.
_patch_stub_handler.is_patch_stub = True  # type: ignore[attr-defined]


def _env_flag(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_bootstrap_fast(
    bootstrap_fast: bool | None,
) -> tuple[bool, bool, bool]:
    bootstrap_context = any(
        _env_flag(env) for env in ("MENACE_BOOTSTRAP_FAST", "MENACE_BOOTSTRAP_MODE", "MENACE_BOOTSTRAP")
    )
    if bootstrap_fast is not None:
        return bootstrap_fast, bootstrap_context, False
    if bootstrap_context:
        return True, bootstrap_context, True
    return False, bootstrap_context, True


def load_handlers(
    *, bootstrap_fast: bool | None = None
) -> Dict[str, Callable[[Dict[str, any]], list[float]]]:
    """Instantiate all registered vectorisers and return transform callables."""

    handlers: Dict[str, Callable[[Dict[str, any]], list[float]]] = {}
    bootstrap_fast, bootstrap_context, defaulted_fast = _resolve_bootstrap_fast(
        bootstrap_fast
    )
    start = time.perf_counter()
    logger.debug(
        "vector_registry.load_handlers.start",
        extra={"registered": len(_VECTOR_REGISTRY)},
    )
    if bootstrap_context and bootstrap_fast:
        logger.info(
            "vector_registry.bootstrap_fast.enabled",
            extra={
                "bootstrap_context": True,
                "bootstrap_fast": bootstrap_fast,
                "defaulted": defaulted_fast,
            },
        )
    for kind, (mod_name, cls_name, _, _) in _VECTOR_REGISTRY.items():
        handler_start = time.perf_counter()
        logger.debug(
            "vector_registry.handler.init",
            extra={
                "kind": kind,
                "module_name": mod_name,
                "class_name": cls_name,
            },
        )
        if bootstrap_fast and kind == "patch":
            logger.info(
                "vector_registry.handler.deferred",
                extra={
                    "kind": kind,
                    "reason": "bootstrap_fast",
                    "bootstrap_context": bootstrap_context,
                },
            )
            handlers[kind] = _patch_stub_handler
            continue
        try:
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, cls_name)
            kwargs = {}
            if _accepts_bootstrap_fast(cls):
                kwargs["bootstrap_fast"] = bootstrap_fast
            inst = cls(**kwargs)
            handlers[kind] = inst.transform
            logger.info(
                "vector_registry.handler.loaded kind=%s duration=%.6fs",
                kind,
                time.perf_counter() - handler_start,
                extra={
                    "kind": kind,
                    "duration_s": round(time.perf_counter() - handler_start, 6),
                },
            )
        except Exception:  # pragma: no cover - best effort to skip bad entries
            logger.exception(
                "vector_registry.handler.error kind=%s duration=%.6fs",
                kind,
                time.perf_counter() - handler_start,
                extra={
                    "kind": kind,
                    "duration_s": round(time.perf_counter() - handler_start, 6),
                },
            )
            continue
    logger.info(
        "vector_registry.load_handlers.complete count=%s duration=%.6fs",
        len(handlers),
        time.perf_counter() - start,
        extra={
            "handler_count": len(handlers),
            "duration_s": round(time.perf_counter() - start, 6),
        },
    )
    return handlers


def get_db_registry() -> Dict[str, Tuple[str, str]]:
    """Return mapping of kind -> (db module, db class) for backfills."""

    mapping: Dict[str, Tuple[str, str]] = {}
    for kind, (_, _, mod, cls) in _VECTOR_REGISTRY.items():
        if mod and cls:
            mapping[kind] = (mod, cls)
    return mapping


# ---------------------------------------------------------------------------
def _discover_vectorizers() -> None:
    """Automatically register vectorizers within :mod:`vector_service`.

    Any submodule whose name ends with ``"vectorizer"`` is imported and
    inspected for classes whose names end with ``"Vectorizer"``.  These are
    registered using the class name prefix (lower‑cased) as the registry key.
    ``DB_MODULE`` and ``DB_CLASS`` attributes, either on the module or the
    class, are used when present to also register the corresponding database
    implementation.  Existing registrations take precedence and are not
    overridden.
    """

    pkg_name = __name__.rsplit(".", 1)[0]
    try:
        pkg = importlib.import_module(pkg_name)
    except Exception:  # pragma: no cover - defensive
        return

    for modinfo in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        # Only consider modules that look like vectorizers to avoid importing
        # heavy dependencies unnecessarily.
        if not modinfo.name.endswith("vectorizer"):
            continue
        try:
            mod = importlib.import_module(modinfo.name)
        except Exception:  # pragma: no cover - best effort
            continue
        for attr, obj in inspect.getmembers(mod, inspect.isclass):
            if not attr.endswith("Vectorizer"):
                continue
            kind = attr[:-10].lower()
            if kind in _VECTOR_REGISTRY:
                continue
            db_module = getattr(obj, "DB_MODULE", getattr(mod, "DB_MODULE", None))
            db_class = getattr(obj, "DB_CLASS", getattr(mod, "DB_CLASS", None))
            register_vectorizer(
                kind,
                modinfo.name,
                attr,
                db_module=db_module,
                db_class=db_class,
            )


# Register built-in vectorisers and their database counterparts.
register_vectorizer("action", "action_vectorizer", "ActionVectorizer")
register_vectorizer(
    "bot",
    "bot_vectorizer",
    "BotVectorizer",
    db_module="bot_database",
    db_class="BotDB",
)
register_vectorizer(
    "workflow",
    "workflow_vectorizer",
    "WorkflowVectorizer",
    db_module="task_handoff_bot",
    db_class="WorkflowDB",
)
register_vectorizer(
    "enhancement",
    "enhancement_vectorizer",
    "EnhancementVectorizer",
    db_module="chatgpt_enhancement_bot",
    db_class="EnhancementDB",
)
register_vectorizer(
    "error",
    "error_vectorizer",
    "ErrorVectorizer",
    db_module="error_bot",
    db_class="ErrorDB",
)
register_vectorizer(
    "information",
    "information_vectorizer",
    "InformationVectorizer",
    db_module="information_db",
    db_class="InformationDB",
)
register_vectorizer(
    "code",
    "code_vectorizer",
    "CodeVectorizer",
    db_module="code_database",
    db_class="CodeDB",
)
register_vectorizer(
    "discrepancy",
    "discrepancy_vectorizer",
    "DiscrepancyVectorizer",
    db_module="discrepancy_db",
    db_class="DiscrepancyDB",
)
register_vectorizer(
    "failure",
    "failure_vectorizer",
    "FailureVectorizer",
    db_module="failure_learning_system",
    db_class="FailureDB",
)
register_vectorizer(
    "patch",
    "vector_service.patch_vectorizer",
    "PatchVectorizer",
    db_module="vector_service.patch_vectorizer",
    db_class="PatchVectorizer",
)
register_vectorizer(
    "research",
    "research_vectorizer",
    "ResearchVectorizer",
    db_module="research_aggregator_bot",
    db_class="InfoDB",
)
register_vectorizer(
    "resource",
    "resource_vectorizer",
    "ResourceVectorizer",
    db_module="resources_bot",
    db_class="ROIHistoryDB",
)
register_vectorizer(
    "intent",
    "intent_vectorizer",
    "IntentVectorizer",
    db_module="intent_db",
    db_class="IntentDB",
)

# Discover any additional vectorisers packaged under vector_service.*
if os.getenv("VECTOR_SERVICE_SKIP_DISCOVERY") != "1":
    _discover_vectorizers()
