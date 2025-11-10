"""Public interface for the :mod:`vector_service` package.

This package provides the canonical vector retrieval service.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, Dict

# ``vector_service`` can be imported in environments where ``menace_sandbox``
# has not yet registered ``import_compat`` as a package attribute (for
# example when running the CLI from a flat layout).  Prefer the relative
# import to avoid relying on that side effect but keep the absolute variant as
# a fallback for existing runtimes.
_BOOTSTRAP_ERROR: ModuleNotFoundError | None = None


def _define_fallback_loader(
    bootstrap_error: ModuleNotFoundError | None,
) -> None:
    """Register the filesystem based ``load_internal`` shim."""

    global load_internal, _BOOTSTRAP_ERROR

    if bootstrap_error is None:
        bootstrap_error = ModuleNotFoundError(
            "import_compat is unavailable and vector_service cannot bootstrap",
        )

    _BOOTSTRAP_ERROR = bootstrap_error

    def load_internal(name: str):
        module_name = name.lstrip(".")
        if module_name.startswith(f"{__name__}."):
            module_name = module_name[len(__name__) + 1 :]
        if module_name.startswith("menace_sandbox."):
            module_name = module_name[len("menace_sandbox.") :]

        qualified = (
            f"{__name__}.{module_name}"
            if not module_name.startswith("vector_service")
            else module_name
        )
        cached = sys.modules.get(qualified) or sys.modules.get(module_name)
        if cached is not None:
            return cached

        root = Path(__file__).resolve().parent
        repo_root = root.parent
        parts = Path(*module_name.split("."))
        for base in (root, repo_root):
            module_path = base / parts
            candidates = [module_path.with_suffix(".py"), module_path / "__init__.py"]
            for candidate in candidates:
                if not candidate.exists():
                    continue
                spec = importlib.util.spec_from_file_location(qualified, candidate)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                sys.modules[qualified] = module
                spec.loader.exec_module(module)
                return module
        raise _BOOTSTRAP_ERROR


try:
    from .. import import_compat as _import_compat
except ImportError:  # pragma: no cover - defensive fallback
    import importlib

    try:
        _import_compat = importlib.import_module("import_compat")
    except ModuleNotFoundError:
        _import_compat = None


if _import_compat is None:
    _define_fallback_loader(None)
else:
    try:  # pragma: no cover - optional dependencies may prevent full bootstrap
        _import_compat.bootstrap(__name__, __file__)
    except ModuleNotFoundError as bootstrap_error:  # pragma: no cover - fallback
        _define_fallback_loader(bootstrap_error)
    else:  # pragma: no cover - fully provisioned environment
        load_internal = _import_compat.load_internal


class _Stub:  # pragma: no cover - simple callable placeholder
    def __init__(self, *args, **kwargs):
        pass


def _noop(*args, **kwargs):  # pragma: no cover - trivial fallback
    return False


class _SimpleSharedVectorService:
    """Lightweight fallback implementation used when vectorizer is unavailable."""

    def __init__(self, embedder: Any | None = None, vector_store: Any | None = None):
        self.embedder = embedder or _Stub()
        self.vector_store = vector_store

    def _ensure_vector(self, kind: str, payload: Dict[str, Any]) -> list[float]:
        if kind == "bot":
            try:  # pragma: no cover - best effort import
                from bot_vectorizer import BotVectorizer  # type: ignore
            except Exception:
                return [0.0]
            return [0.0] * BotVectorizer().dim

        text = ""
        if kind == "text":
            text = str(payload.get("text", ""))
        else:
            text = " ".join(
                str(value)
                for key, value in payload.items()
                if isinstance(value, str) and key != "text"
            )

        if hasattr(self.embedder, "encode"):
            try:
                vecs = self.embedder.encode([text])
                vec = vecs[0]
                if hasattr(vec, "tolist"):
                    vec = vec.tolist()
                return [float(x) for x in vec]
            except Exception:
                pass
        return [0.0]

    def vectorise(self, kind: str, payload: Dict[str, Any]) -> list[float]:
        return self._ensure_vector(kind, payload)

    def vectorise_and_store(
        self, kind: str, record_id: str, payload: Dict[str, Any]
    ) -> list[float]:
        vec = self.vectorise(kind, payload)
        if self.vector_store is not None:
            try:
                self.vector_store.add(
                    kind, record_id, vec, metadata=payload
                )
            except Exception:  # pragma: no cover - defensive fallback
                pass
        return vec


# ``Retriever`` historically provided ``FallbackResult`` so we default to the
# lightweight implementations first and upgrade them when the real modules are
# available.  This ensures partial imports do not mask functionality unrelated to
# a missing heavy dependency (for example ``transformers``).
class FallbackResult(list):  # pragma: no cover - used when retriever unavailable
    pass


class VectorServiceError(Exception):  # pragma: no cover - default error type
    pass


RateLimitError = MalformedPromptError = VectorServiceError

Retriever = PatchLogger = CognitionLayer = EmbeddingBackfill = ContextBuilder = _Stub  # type: ignore
SharedVectorService: type[_SimpleSharedVectorService] | type[_Stub] = _SimpleSharedVectorService
StackDatasetStreamer = _Stub  # type: ignore
StackRetriever = _Stub  # type: ignore
ensure_stack_background = _noop
run_stack_ingestion_async = _Stub  # type: ignore

try:  # pragma: no cover - upgrade default errors when available
    from .exceptions import (
        VectorServiceError as _VectorServiceError,
        RateLimitError as _RateLimitError,
        MalformedPromptError as _MalformedPromptError,
    )
except Exception:
    pass
else:
    VectorServiceError = _VectorServiceError
    RateLimitError = _RateLimitError
    MalformedPromptError = _MalformedPromptError

try:  # pragma: no cover - optional heavy dependency
    from .retriever import Retriever as _Retriever, FallbackResult as _FallbackResult
except Exception:
    pass
else:
    Retriever = _Retriever
    FallbackResult = _FallbackResult

try:  # pragma: no cover - optional heavy dependency
    from .patch_logger import PatchLogger as _PatchLogger
except Exception:
    pass
else:
    PatchLogger = _PatchLogger

try:  # pragma: no cover - optional heavy dependency
    from .cognition_layer import CognitionLayer as _CognitionLayer
except Exception:
    pass
else:
    CognitionLayer = _CognitionLayer

try:  # pragma: no cover - optional heavy dependency
    _embedding_backfill_module = load_internal("vector_service.embedding_backfill")
except ModuleNotFoundError as exc:
    if getattr(exc, "name", None) not in {
        "vector_service.embedding_backfill",
        "menace_sandbox.vector_service.embedding_backfill",
    }:
        raise
else:
    EmbeddingBackfill = _embedding_backfill_module.EmbeddingBackfill  # type: ignore[attr-defined]

try:  # pragma: no cover - optional heavy dependency
    from .vectorizer import SharedVectorService as _SharedVectorService
except Exception:
    pass
else:
    SharedVectorService = _SharedVectorService

try:  # pragma: no cover - optional heavy dependency
    from .stack_ingestion import (
        StackDatasetStreamer as _StackDatasetStreamer,
        ensure_background_task as _ensure_stack_background,
        run_stack_ingestion_async as _run_stack_ingestion_async,
    )
except Exception:
    pass
else:
    StackDatasetStreamer = _StackDatasetStreamer
    ensure_stack_background = _ensure_stack_background
    run_stack_ingestion_async = _run_stack_ingestion_async

try:  # pragma: no cover - optional heavy dependency
    from .stack_retriever import StackRetriever as _StackRetriever
except Exception:
    pass
else:
    StackRetriever = _StackRetriever

try:  # pragma: no cover - prefer lightweight stack retriever facade when available
    from .retriever import StackRetriever as _StackContextRetriever
except Exception:
    pass
else:
    StackRetriever = _StackContextRetriever

try:  # pragma: no cover - optional heavy dependency
    from .context_builder import ContextBuilder as _ContextBuilder
except Exception:
    pass
else:
    ContextBuilder = _ContextBuilder


class ErrorResult(Exception):
    """Fallback error result used when retriever returns an error."""

    pass

try:  # pragma: no cover - optional dependency used in tests
    _embeddable_db_module = load_internal("embeddable_db_mixin")
except ModuleNotFoundError as exc:  # pragma: no cover - fallback when module missing
    if getattr(exc, "name", None) not in {
        "embeddable_db_mixin",
        "menace_sandbox.embeddable_db_mixin",
    }:
        raise

    class _FallbackEmbeddableDBMixin:
        """Stub mixin used when the vector service cannot load the real one."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            self._embeddings_enabled = False
            self.event_bus = kwargs.get("event_bus")

            try:
                mro = type(self).__mro__
                next_cls = mro[mro.index(_FallbackEmbeddableDBMixin) + 1]
            except (ValueError, IndexError):
                next_cls = None

            if next_cls in {None, object}:
                super().__init__()
            else:
                super().__init__(*args, **kwargs)

        def backfill_embeddings(self) -> list[object]:  # pragma: no cover - simple stub
            return []

        def search_by_vector(
            self, *args: object, **kwargs: object
        ) -> list[object]:  # pragma: no cover - simple stub
            return []

    EmbeddableDBMixin = _FallbackEmbeddableDBMixin  # type: ignore

    _fallback_logger = logging.getLogger(__name__)

    def safe_super_init(
        cls: type, instance: object, *args: object, **kwargs: object
    ) -> None:
        """Fallback cooperative init helper when embeddings are disabled."""

        try:
            mro = type(instance).__mro__
            next_cls = mro[mro.index(cls) + 1]
        except (ValueError, IndexError):
            next_cls = None

        if next_cls is object:
            if kwargs:
                _fallback_logger.debug(
                    "[vector-service] Dropping kwargs for object.__init__: cls=%s kwargs=%s",
                    cls.__name__,
                    list(kwargs),
                )
            if args:
                _fallback_logger.debug(
                    "[vector-service] Dropping args for object.__init__: cls=%s args=%s",
                    cls.__name__,
                    args,
                )
            super(cls, instance).__init__()
            return

        super(cls, instance).__init__(*args, **kwargs)

    def safe_super_init_or_warn(
        cls: type,
        instance: object,
        *args: object,
        logger: logging.Logger | None = None,
        **kwargs: object,
    ) -> None:
        target_logger = logger or _fallback_logger
        try:
            mro = type(instance).__mro__
            next_cls = mro[mro.index(cls) + 1]
        except (ValueError, IndexError):
            next_cls = None

        if next_cls is object and (args or kwargs):
            target_logger.debug(
                "[vector-service] Dropping cooperative args for %s -> object.__init__: args=%s kwargs=%s",
                cls.__name__,
                args,
                kwargs,
            )

        safe_super_init(cls, instance, *args, **kwargs)
else:
    if hasattr(_embeddable_db_module, "EmbeddableDBMixin"):
        EmbeddableDBMixin = _embeddable_db_module.EmbeddableDBMixin  # type: ignore[attr-defined]
    else:  # pragma: no cover - degrade when mixin unavailable

        class EmbeddableDBMixin:  # type: ignore
            def __init__(self, *args: object, **kwargs: object) -> None:
                self.event_bus = kwargs.get("event_bus")

                try:
                    mro = type(self).__mro__
                    next_cls = mro[mro.index(EmbeddableDBMixin) + 1]
                except (ValueError, IndexError):
                    next_cls = None

                if next_cls in {None, object}:
                    super().__init__()
                else:
                    super().__init__(*args, **kwargs)

    if hasattr(_embeddable_db_module, "safe_super_init"):
        safe_super_init = _embeddable_db_module.safe_super_init  # type: ignore[attr-defined]
    else:  # pragma: no cover - degrade gracefully when helper missing

        _fallback_logger = logging.getLogger(__name__)

        def safe_super_init(
            cls: type, instance: object, *args: object, **kwargs: object
        ) -> None:
            try:
                mro = type(instance).__mro__
                next_cls = mro[mro.index(cls) + 1]
            except (ValueError, IndexError):
                next_cls = None

            if next_cls is object:
                if kwargs:
                    _fallback_logger.debug(
                        "[vector-service] Dropping kwargs for object.__init__: cls=%s kwargs=%s",
                        cls.__name__,
                        list(kwargs),
                    )
                if args:
                    _fallback_logger.debug(
                        "[vector-service] Dropping args for object.__init__: cls=%s args=%s",
                        cls.__name__,
                        args,
                    )
                super(cls, instance).__init__()
                return

            super(cls, instance).__init__(*args, **kwargs)

    if hasattr(_embeddable_db_module, "safe_super_init_or_warn"):
        safe_super_init_or_warn = _embeddable_db_module.safe_super_init_or_warn  # type: ignore[attr-defined]
    else:  # pragma: no cover - degrade gracefully when helper missing

        _fallback_logger = logging.getLogger(__name__)

        def safe_super_init_or_warn(
            cls: type,
            instance: object,
            *args: object,
            logger: logging.Logger | None = None,
            **kwargs: object,
        ) -> None:
            target_logger = logger or _fallback_logger
            try:
                mro = type(instance).__mro__
                next_cls = mro[mro.index(cls) + 1]
            except (ValueError, IndexError):
                next_cls = None

            if next_cls is object and (args or kwargs):
                target_logger.debug(
                    "[vector-service] Dropping cooperative args for %s -> object.__init__: args=%s kwargs=%s",
                    cls.__name__,
                    args,
                    kwargs,
                )

            safe_super_init(cls, instance, *args, **kwargs)

    sys.modules.setdefault("embeddable_db_mixin", _embeddable_db_module)
    sys.modules.setdefault(
        "menace_sandbox.embeddable_db_mixin", _embeddable_db_module
    )


__all__ = [
    "Retriever",
    "FallbackResult",
    "PatchLogger",
    "CognitionLayer",
    "EmbeddingBackfill",
    "SharedVectorService",
    "ContextBuilder",
    "StackDatasetStreamer",
    "StackRetriever",
    "ensure_stack_background",
    "run_stack_ingestion_async",
    "EmbeddableDBMixin",
    "VectorServiceError",
    "RateLimitError",
    "MalformedPromptError",
    "ErrorResult",
    "safe_super_init",
    "safe_super_init_or_warn",
]
