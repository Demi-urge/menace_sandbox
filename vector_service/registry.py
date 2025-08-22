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


def load_handlers() -> Dict[str, Callable[[Dict[str, any]], list[float]]]:
    """Instantiate all registered vectorisers and return transform callables."""

    handlers: Dict[str, Callable[[Dict[str, any]], list[float]]] = {}
    for kind, (mod_name, cls_name, _, _) in _VECTOR_REGISTRY.items():
        try:
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, cls_name)
            inst = cls()
            handlers[kind] = inst.transform
        except Exception:  # pragma: no cover - best effort to skip bad entries
            continue
    return handlers


def get_db_registry() -> Dict[str, Tuple[str, str]]:
    """Return mapping of kind -> (db module, db class) for backfills."""

    mapping: Dict[str, Tuple[str, str]] = {}
    for kind, (_, _, mod, cls) in _VECTOR_REGISTRY.items():
        if mod and cls:
            mapping[kind] = (mod, cls)
    return mapping


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
