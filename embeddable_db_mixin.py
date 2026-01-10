"""Mixin providing embedding storage and vector search backends.

This module offers :class:`EmbeddableDBMixin` which can be mixed into a
class managing a SQLite database.  The mixin stores embedding vectors in an
Annoy or FAISS index on disk and keeps companion metadata in a JSON file.  A
 lazily loaded `SentenceTransformer` model is provided for text-to-vector
encoding, allowing subclasses to embed arbitrary records.

Subclasses must provide a ``self.conn`` database connection and override
:meth:`vector` to return an embedding for a record.  To support
:meth:`backfill_embeddings`, subclasses should also implement
:meth:`iter_records` yielding ``(record_id, record, kind)`` tuples.
"""

from __future__ import annotations

_DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_canonical_model_id_fn: "Callable[[str | None], str] | None" = None


def _normalise_model_name(model_name: str | None) -> str:
    """Return ``model_name`` with the canonical SentenceTransformer prefix."""

    if _canonical_model_id_fn is not None:
        return _canonical_model_id_fn(model_name)
    name = (model_name or "").strip()
    if not name:
        return _DEFAULT_MODEL_NAME
    if "/" not in name:
        return f"sentence-transformers/{name}"
    return name

import importlib.util
import os
import sys
import types
from pathlib import Path

_HELPER_NAME = "import_compat"
_PACKAGE_NAME = "menace_sandbox"

try:  # pragma: no cover - prefer package import when available
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

try:
    import_compat.bootstrap(__name__, __file__)
except ModuleNotFoundError as bootstrap_error:  # pragma: no cover - optional deps

    def load_internal(name: str):
        """Best effort loader that avoids importing the heavy package root."""

        qualified = f"{_PACKAGE_NAME}.{name}"
        cached = sys.modules.get(qualified) or sys.modules.get(name)
        if cached is not None:
            return cached

        module_path = Path(__file__).resolve().parent / Path(*name.split("."))
        candidates = [module_path.with_suffix(".py"), module_path / "__init__.py"]
        for candidate in candidates:
            if not candidate.exists():
                continue
            spec = importlib.util.spec_from_file_location(qualified, candidate)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            sys.modules[qualified] = module
            spec.loader.exec_module(module)
            return module

        raise bootstrap_error

else:  # pragma: no cover - full environment
    load_internal = import_compat.load_internal

from dataclasses import dataclass
from datetime import datetime
from time import perf_counter
from typing import Any, Callable, Dict, Iterator, List, Sequence, Tuple
import hashlib
import json
import logging
import re

try:
    _secret_redactor = load_internal("security.secret_redactor")
    redact = _secret_redactor.redact
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    redact = lambda text: text  # type: ignore

try:
    _semantic_diff = load_internal("analysis.semantic_diff_filter")
    find_semantic_risks = _semantic_diff.find_semantic_risks
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    find_semantic_risks = lambda sentences: []  # type: ignore

try:
    _governed_embeddings = load_internal("governed_embeddings")
    governed_embed = _governed_embeddings.governed_embed
    _DEFAULT_MODEL_NAME = getattr(
        _governed_embeddings,
        "DEFAULT_SENTENCE_TRANSFORMER_MODEL",
        _DEFAULT_MODEL_NAME,
    )
    _SENTENCE_TRANSFORMER_DEVICE = getattr(
        _governed_embeddings, "SENTENCE_TRANSFORMER_DEVICE", "cpu"
    )
    _canonical_model_id_fn = getattr(
        _governed_embeddings,
        "canonical_model_id",
        None,
    )
    _initialise_sentence_transformer = getattr(
        _governed_embeddings,
        "initialise_sentence_transformer",
        None,
    )
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    governed_embed = lambda text, model=None: []  # type: ignore
    _SENTENCE_TRANSFORMER_DEVICE = "cpu"
    _initialise_sentence_transformer = None

try:
    _chunking = load_internal("chunking")
    split_into_chunks = _chunking.split_into_chunks
    summarize_snippet = _chunking.summarize_snippet
except ModuleNotFoundError:  # pragma: no cover - optional dependency

    class _Chunk:
        def __init__(self, text: str) -> None:
            self.text = text

    def split_into_chunks(text: str, size: int) -> list[_Chunk]:  # type: ignore
        return [_Chunk(text)] if text else []

    def summarize_snippet(text: str, *_args, **_kwargs) -> str:  # type: ignore
        return text

try:
    _text_preprocessor = load_internal("vector_service.text_preprocessor")
    generalise = _text_preprocessor.generalise
    PreprocessingConfig = _text_preprocessor.PreprocessingConfig
except ModuleNotFoundError:  # pragma: no cover - optional dependency

    @dataclass
    class PreprocessingConfig:  # type: ignore
        stop_words: set[str] | None = None
        language: str | None = None
        use_lemmatizer: bool = True
        split_sentences: bool = True
        chunk_size: int = 400
        filter_semantic_risks: bool = True

    def generalise(text: str, *, config=None, db_key=None):  # type: ignore
        return text

# Lightweight license detection based on SPDX‑style fingerprints.  This avoids
# embedding content that is under GPL or non‑commercial restrictions.
try:
    _license_fingerprint = load_internal("compliance.license_fingerprint")
    license_check = _license_fingerprint.check
    license_fingerprint = _license_fingerprint.fingerprint
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    license_check = lambda *_args, **_kwargs: True  # type: ignore
    license_fingerprint = lambda text: ""  # type: ignore

try:  # pragma: no cover - optional dependency
    from annoy import AnnoyIndex
except Exception:  # pragma: no cover - Annoy not installed
    AnnoyIndex = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import faiss  # type: ignore
except Exception:  # pragma: no cover - FAISS not installed
    faiss = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - NumPy not installed
    np = None  # type: ignore


def safe_super_init(cls: type, instance: Any, *args: Any, **kwargs: Any) -> None:
    """Cooperatively call ``super().__init__`` while guarding ``object``."""

    if not isinstance(instance, cls):
        if args or kwargs:
            print(
                "[cooperative-init] Skipping super() for"
                f" {cls.__name__} because {type(instance).__name__}"
                " is not a subtype; dropping args/kwargs to avoid TypeError"
            )
        return

    try:
        mro = type(instance).__mro__
        next_cls = mro[mro.index(cls) + 1]
    except (ValueError, IndexError):
        next_cls = None

    if next_cls is object:
        if kwargs:
            print(f"[trace] Dropping kwargs at end of MRO: {kwargs}")
        if args:
            print(f"[trace] Dropping args at end of MRO: {args}")
        super(cls, instance).__init__()
        return

    super(cls, instance).__init__(*args, **kwargs)


def safe_super_init_or_warn(
    cls: type,
    instance: Any,
    *args: Any,
    logger: logging.Logger | None = None,
    **kwargs: Any,
) -> None:
    """Invoke :func:`safe_super_init` while logging cooperative fallbacks."""

    try:
        mro = type(instance).__mro__
        next_cls = mro[mro.index(cls) + 1]
    except (ValueError, IndexError):
        next_cls = None

    if next_cls is object and (args or kwargs):
        message = (
            f"[cooperative-init] Dropping args={args!r} kwargs={kwargs!r} for "
            f"{cls.__name__} -> object.__init__"
        )
        if logger is not None:
            logger.debug(message)
        else:
            print(message)

    safe_super_init(cls, instance, *args, **kwargs)

try:
    _metrics_exporter = load_internal("metrics_exporter")
    _EMBED_STORE_LAST = _metrics_exporter.embedding_store_latency_seconds
    _EMBED_STORE_TOTAL = _metrics_exporter.embedding_store_seconds_total
    _EMBED_STALE = _metrics_exporter.embedding_stale_cost_seconds
    _EMBED_TOKENS = _metrics_exporter.embedding_tokens_total
    _EMBED_WALL_TOTAL = _metrics_exporter.embedding_wall_seconds_total
    _EMBED_WALL_LAST = _metrics_exporter.embedding_wall_time_seconds
except ModuleNotFoundError:  # pragma: no cover - optional dependency

    class _MetricStub:
        def inc(self, *_args, **_kwargs):
            return None

        def set(self, *_args, **_kwargs):
            return None

    _EMBED_STORE_LAST = _EMBED_STORE_TOTAL = _EMBED_STALE = _MetricStub()
    _EMBED_TOKENS = _EMBED_WALL_TOTAL = _EMBED_WALL_LAST = _MetricStub()

try:
    _vector_metrics_db = load_internal("vector_metrics_db")
    VectorMetricsDB = _vector_metrics_db.VectorMetricsDB
    get_bootstrap_vector_metrics_db = _vector_metrics_db.get_bootstrap_vector_metrics_db
    resolve_vector_bootstrap_flags = _vector_metrics_db.resolve_vector_bootstrap_flags
except (ModuleNotFoundError, AttributeError):  # pragma: no cover - optional dependency

    class VectorMetricsDB:  # type: ignore
        def log_embedding(self, *args, **kwargs):
            return None

    get_bootstrap_vector_metrics_db = None  # type: ignore
    resolve_vector_bootstrap_flags = None  # type: ignore

try:
    _embedding_stats_db = load_internal("embedding_stats_db")
    EmbeddingStatsDB = _embedding_stats_db.EmbeddingStatsDB
except (ModuleNotFoundError, AttributeError):  # pragma: no cover - optional dependency

    class EmbeddingStatsDB:  # type: ignore
        def __init__(self, *_args, **_kwargs):
            pass

        def log(self, *args, **kwargs):
            return None

logger = logging.getLogger(__name__)


class _MetricsDBStub:
    """Fallback implementation when :mod:`data_bot` is unavailable."""

    def __init__(self, *_args, **_kwargs) -> None:  # pragma: no cover - trivial
        return None

    def log_embedding_staleness(self, *_args, **_kwargs) -> None:  # pragma: no cover
        return None


_METRICS_DB_CLS: type[Any] | None = None
# Allow tests to inject a custom MetricsDB implementation.
MetricsDB: type[Any] | None = None


class _VectorMetricsWarmupPlaceholder:
    """Minimal in-memory placeholder for warmup stubs.

    The placeholder captures pending configuration so callers can later
    promote it to a real ``VectorMetricsDB`` without touching disk during
    warmup or bootstrap flows.
    """

    def __init__(self, *, warmup_kwargs: dict[str, Any], stub_trigger: str) -> None:
        self._pending_weights: dict[str, float] = {}
        self._readiness_hook_requested = False
        self._first_write_requested = False
        self._delegate: Any | None = None
        self._activation_kwargs: dict[str, Any] | None = None
        self._activation_factory: type[Any] | None = None
        self._warmup_kwargs = dict(warmup_kwargs)
        self._stub_trigger = stub_trigger

    def log_embedding(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - noop
        if self._delegate is None and self._activation_factory and self._activation_kwargs:
            self._delegate = self.promote(
                self._activation_factory, dict(self._activation_kwargs)
            )
        if self._delegate is not None:
            return self._delegate.log_embedding(*args, **kwargs)
        return None

    def activate_on_first_write(self) -> None:
        self._first_write_requested = True

    def register_readiness_hook(self, hook: Any | None = None) -> None:
        # The concrete DB registers its own readiness hook; we only need to
        # remember that the request happened.
        self._readiness_hook_requested = True
        if self._delegate is None and self._activation_factory and self._activation_kwargs:
            self._delegate = self.promote(
                self._activation_factory, dict(self._activation_kwargs)
            )
        if callable(hook):  # pragma: no cover - optional passthrough
            try:
                if self._delegate is not None:
                    delegate_hook = getattr(self._delegate, "register_readiness_hook", None)
                    if callable(delegate_hook):
                        delegate_hook(hook)
                    else:
                        hook()
                else:
                    hook()
            except Exception:
                logger.debug("vector metrics warmup readiness hook failed", exc_info=True)

    def set_db_weights(self, weights: Mapping[str, float]) -> None:
        self._pending_weights.update({str(k): float(v) for k, v in weights.items()})

    def prepare_promotion(
        self, *, factory: type[Any], activation_kwargs: dict[str, Any]
    ) -> None:
        if self._delegate is not None:
            return
        self._activation_factory = factory
        self._activation_kwargs = dict(activation_kwargs)
        logger.info(
            "embeddable_db_mixin.vector_metrics.warmup_promotion_armed",
            extra={
                "stub_trigger": self._stub_trigger,
                "activation_keys": sorted(self._activation_kwargs),
            },
        )

    def promote(self, factory: type[Any], activation_kwargs: dict[str, Any]) -> Any:
        combined_kwargs = dict(self._warmup_kwargs)
        combined_kwargs.update(activation_kwargs)
        delegate = factory(**combined_kwargs)
        if self._pending_weights:
            setter = getattr(delegate, "set_db_weights", None)
            if callable(setter):
                try:
                    setter(dict(self._pending_weights))
                except Exception:  # pragma: no cover - defensive logging
                    logger.debug("vector metrics warmup weight replay failed", exc_info=True)
        if self._first_write_requested and hasattr(delegate, "activate_on_first_write"):
            delegate.activate_on_first_write()
        if self._readiness_hook_requested:
            hook = getattr(delegate, "register_readiness_hook", None)
            if callable(hook):
                hook()
        logger.info(
            "embeddable_db_mixin.vector_metrics.warmup_promoted",
            extra={"stub_trigger": self._stub_trigger},
        )
        self._delegate = delegate
        return delegate


def _resolve_metrics_db() -> type[Any]:
    """Return :class:`data_bot.MetricsDB` without triggering circular imports."""

    global _METRICS_DB_CLS
    if _METRICS_DB_CLS is not None:
        return _METRICS_DB_CLS
    if MetricsDB is not None:
        _METRICS_DB_CLS = MetricsDB
        return _METRICS_DB_CLS

    try:
        module = load_internal("data_bot")
    except ModuleNotFoundError:
        logger.debug("data_bot module unavailable; using stub MetricsDB")
        cls: type[Any] | None = None
    except Exception:  # pragma: no cover - defensive logging
        logger.exception("failed to load data_bot module for MetricsDB")
        cls = None
    else:
        cls = getattr(module, "MetricsDB", None)
        if cls is None:
            logger.debug("data_bot.MetricsDB missing; using stub implementation")

    if cls is None:
        cls = _MetricsDBStub

    _METRICS_DB_CLS = cls
    globals()["MetricsDB"] = cls
    return cls


_VEC_METRICS: VectorMetricsDB | None = None
_EMBED_STATS_DB = EmbeddingStatsDB("metrics.db")


def _vector_metrics_db(
    *,
    bootstrap_fast: bool | None = None,
    warmup: bool | None = None,
    warmup_lite: bool | None = None,
    vector_warmup: bool | None = None,
) -> VectorMetricsDB | None:
    global _VEC_METRICS
    warmup_kwargs = {}
    callsite_requested = bool(
        (bootstrap_fast is True)
        or (warmup is True)
        or (warmup_lite is True)
    )
    warmup_mode = bool(warmup)
    vector_warmup_flag = bool(
        vector_warmup
        or os.getenv("VECTOR_WARMUP")
        or os.getenv("VECTOR_SERVICE_WARMUP")
    )
    warmup_lite_flag = bool(warmup_lite)
    env_requested = False
    bootstrap_env = False
    bootstrap_fast_flag = bool(bootstrap_fast)
    bootstrap_state_active = False
    warmup_state_flag = False

    if resolve_vector_bootstrap_flags is not None:
        (
            bootstrap_fast_flag,
            warmup_mode,
            env_requested,
            bootstrap_env,
        ) = resolve_vector_bootstrap_flags(
            bootstrap_fast=bootstrap_fast, warmup=warmup
        )
    else:  # pragma: no cover - fallback when optional helper unavailable
        menace_bootstrap = any(
            os.getenv(flag)
            for flag in (
                "MENACE_BOOTSTRAP",
                "MENACE_BOOTSTRAP_MODE",
                "MENACE_BOOTSTRAP_FAST",
            )
        )
        bootstrap_env = bool(menace_bootstrap)
        env_requested = bool(menace_bootstrap or os.getenv("VECTOR_METRICS_WARMUP"))
        try:
            cbi = load_internal("coding_bot_interface")
        except Exception:
            cbi = None
        state = getattr(cbi, "_BOOTSTRAP_STATE", None)
        bootstrap_state_active = bool(getattr(state, "depth", 0) or state)
        warmup_state_flag = bool(getattr(state, "warmup_lite", False))

    warmup_stub_requested = bool(
        env_requested
        or bootstrap_env
        or bootstrap_fast_flag
        or warmup_state_flag
        or vector_warmup_flag
        or bootstrap_state_active
    )
    warmup_mode = bool(
        warmup_mode or warmup_lite_flag or vector_warmup_flag or warmup_state_flag
    )

    if _VEC_METRICS is not None:
        if isinstance(_VEC_METRICS, _VectorMetricsWarmupPlaceholder):
            if warmup_stub_requested or warmup_mode:
                return _VEC_METRICS

            activation_kwargs = {
                "bootstrap_fast": bootstrap_fast_flag,
                "warmup": False,
                "ensure_exists": True,
                "read_only": False,
            }
            _VEC_METRICS.prepare_promotion(
                factory=VectorMetricsDB, activation_kwargs=activation_kwargs
            )
            logger.info(
                "embeddable_db_mixin.vector_metrics.warmup_promotion_delayed",
                extra={
                    "callsite_requested": callsite_requested,
                    "env_requested": env_requested,
                    "bootstrap_env": bootstrap_env,
                },
            )
            return _VEC_METRICS
        return _VEC_METRICS

    if VectorMetricsDB is None:
        return None

    warmup_mode = bool(
        warmup_mode or warmup_lite_flag or vector_warmup_flag or warmup_state_flag
    )
    warmup_stub = bool(
        warmup_mode
        or env_requested
        or bootstrap_env
        or bootstrap_fast_flag
        or vector_warmup_flag
        or bootstrap_state_active
    )
    warmup_kwargs = {
        "bootstrap_fast": bootstrap_fast_flag,
        "warmup": warmup_mode,
    }
    stub_trigger = "+".join(
        [
            trigger
            for trigger, active in (
                ("call", callsite_requested),
                ("env", env_requested or bootstrap_env),
                ("state", bootstrap_state_active or warmup_state_flag),
                ("vector_warmup", vector_warmup_flag),
            )
            if active
        ]
    )
    stub_trigger = stub_trigger or "implicit"

    if warmup_stub and get_bootstrap_vector_metrics_db is not None:
        warmup_kwargs.update({"ensure_exists": False, "read_only": True})
        _VEC_METRICS = get_bootstrap_vector_metrics_db(**warmup_kwargs)
        try:
            _VEC_METRICS.activate_on_first_write()
            readiness_hook = getattr(_VEC_METRICS, "register_readiness_hook", None)
            if callable(readiness_hook):
                readiness_hook()
        except Exception:
            logger.debug("failed to arm lazy activation for vector metrics", exc_info=True)
        logger.info(
            "embeddable_db_mixin.vector_metrics.stubbed",
            extra={
                "bootstrap_fast": bootstrap_fast_flag,
                "warmup_mode": warmup_mode,
                "warmup_lite": warmup_lite_flag,
                "vector_warmup": vector_warmup_flag,
                "env_bootstrap_requested": env_requested,
                "menace_bootstrap": bootstrap_env,
                "lazy_activation": True,
                "stub_trigger": stub_trigger,
            },
        )
        return _VEC_METRICS
    elif warmup_stub:
        warmup_kwargs.update({"ensure_exists": False, "read_only": True})
        placeholder = _VectorMetricsWarmupPlaceholder(
            warmup_kwargs=warmup_kwargs, stub_trigger=stub_trigger
        )
        placeholder.activate_on_first_write()
        placeholder.register_readiness_hook()
        _VEC_METRICS = placeholder
        logger.info(
            "embeddable_db_mixin.vector_metrics.stubbed",
            extra={
                "bootstrap_fast": bootstrap_fast_flag,
                "warmup_mode": warmup_mode,
                "warmup_lite": warmup_lite_flag,
                "vector_warmup": vector_warmup_flag,
                "env_bootstrap_requested": env_requested,
                "menace_bootstrap": bootstrap_env,
                "lazy_activation": True,
                "stub_trigger": stub_trigger,
            },
        )

    _VEC_METRICS = _VEC_METRICS or VectorMetricsDB(**warmup_kwargs)
    return _VEC_METRICS


def log_embedding_metrics(
    db_name: str,
    tokens: int,
    wall_time: float,
    store_latency: float,
    *,
    vector_id: str = "",
) -> None:
    """Log embedding metrics to Prometheus and persistent storage."""

    try:
        _EMBED_TOKENS.inc(tokens)
        _EMBED_WALL_TOTAL.inc(wall_time)
        _EMBED_STORE_TOTAL.inc(store_latency)
        _EMBED_WALL_LAST.set(wall_time)
        _EMBED_STORE_LAST.set(store_latency)
    except Exception:  # pragma: no cover - best effort
        pass
    try:
        _EMBED_STATS_DB.log(
            db_name=db_name,
            tokens=tokens,
            wall_ms=wall_time * 1000,
            store_ms=store_latency * 1000,
        )
        vec_metrics = _vector_metrics_db()
        if vec_metrics is not None:
            vec_metrics.log_embedding(
                db=db_name,
                tokens=tokens,
                wall_time_ms=wall_time * 1000,
                store_time_ms=store_latency * 1000,
                vector_id=vector_id,
            )
    except Exception:  # pragma: no cover - best effort
        logger.exception("failed to persist embedding metrics")


class EmbeddableDBMixin:
    """Add embedding storage and similarity search to a database class."""

    @classmethod
    def default_embedding_paths(cls) -> tuple[Path, Path] | None:
        """Return default index and metadata paths without instantiating."""

        return None

    def __init__(
        self,
        *super_args: Any,
        index_path: str | Path = "embeddings.ann",
        metadata_path: str | Path = "embeddings.json",
        model_name: str = _DEFAULT_MODEL_NAME,
        embedding_version: int = 1,
        backend: str = "annoy",
        event_bus: Any | None = None,
        defer_index_load: bool | None = None,
        **super_kwargs: Any,
    ) -> None:
        """Initialise the mixin while tolerating cooperative super-calls."""

        logger.debug(
            "[trace] MRO chain: %s",
            [cls.__name__ for cls in type(self).__mro__],
        )

        passthrough = dict(super_kwargs)
        if event_bus is not None:
            passthrough.setdefault("event_bus", event_bus)

        bus = passthrough.get("event_bus")
        if bus is not None:
            setattr(self, "event_bus", bus)

        safe_super_init(EmbeddableDBMixin, self, *super_args, **passthrough)

        index_path = Path(index_path)
        metadata_path = Path(metadata_path)
        if metadata_path.name == "embeddings.json" and index_path.name != "embeddings.ann":
            # Derive metadata file alongside the provided index path to avoid
            # cross-database interference when tests supply unique index files
            metadata_path = index_path.with_suffix(".json")
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.model_name = _normalise_model_name(model_name)
        self.embedding_version = embedding_version
        self.backend = backend

        self._model = None
        self._index: Any | None = None
        self._vector_dim = 0
        self._id_map: List[str] = []
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._last_embedding_tokens = 0
        self._last_embedding_time = 0.0
        self._last_chunk_meta: Dict[str, Any] = {}

        if defer_index_load is None:
            bootstrap_env = any(
                str(os.getenv(flag, "")).lower() in {"1", "true", "yes", "on"}
                for flag in (
                    "MENACE_BOOTSTRAP_FAST",
                    "MENACE_BOOTSTRAP_MODE",
                    "MENACE_BOOTSTRAP",
                    "VECTOR_SERVICE_WARMUP",
                )
            )
            defer_index_load = bootstrap_env
        self._defer_index_load = bool(defer_index_load)
        self._index_loaded = False
        if not self._defer_index_load:
            self.load_index()

    # ------------------------------------------------------------------
    # model helpers
    @property
    def model(self):
        """Lazily loaded `SentenceTransformer` instance."""
        if self._model is None:  # pragma: no cover - heavy dependency
            from sentence_transformers import SentenceTransformer
            from huggingface_hub import login
            import os

            login(token=os.getenv("HUGGINGFACE_API_TOKEN"))
            model_name = _normalise_model_name(self.model_name)
            self.model_name = model_name
            if _initialise_sentence_transformer is not None:
                kwargs: dict[str, object] = {}
                if _SENTENCE_TRANSFORMER_DEVICE:
                    kwargs["device"] = _SENTENCE_TRANSFORMER_DEVICE
                self._model = _initialise_sentence_transformer(
                    model_name, **kwargs
                )
            else:
                self._model = SentenceTransformer(
                    model_name, device=_SENTENCE_TRANSFORMER_DEVICE
                )
        return self._model

    def encode_text(self, text: str) -> List[float]:
        """Encode ``text`` using the SentenceTransformer model."""

        self._ensure_index_loaded()
        start = perf_counter()
        vec = governed_embed(text, self.model)
        self._last_embedding_time = perf_counter() - start
        tokens = 0
        if vec is not None:
            try:  # pragma: no cover - optional dependency
                tokenizer = getattr(self.model, "tokenizer", None)
                if tokenizer:
                    tokens = len(tokenizer.encode(redact(text)))
            except Exception:
                tokens = 0
        self._last_embedding_tokens = tokens
        return vec or []

    def _split_and_summarise(
        self,
        text: str,
        *,
        config: "PreprocessingConfig" | None = None,
        db_key: str | None = None,
    ) -> str:
        """Split ``text`` into sentences, filter, chunk and summarise.

        ``config`` may override the behaviour for sentence splitting,
        chunk sizes and semantic risk filtering.  When ``config`` is ``None``
        a configuration registered for ``db_key`` will be used if available.
        The resulting condensed text is returned while ``self._last_chunk_meta``
        records ``chunk_count`` and ``chunk_hashes`` for traceability.
        """

        from vector_service.text_preprocessor import get_config

        cfg = config or get_config(db_key or self.__class__.__name__.lower())

        if not isinstance(text, str):
            self._last_chunk_meta = {"chunk_count": 0, "chunk_hashes": []}
            return text

        if cfg.split_sentences:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        else:
            sentences = [text]
        if not sentences:
            self._last_chunk_meta = {"chunk_count": 0, "chunk_hashes": []}
            return ""

        alerts = find_semantic_risks(sentences) if cfg.filter_semantic_risks else []
        risky = {line for line, _, _ in alerts}
        filtered = [s for s in sentences if s not in risky]
        joined = "\n".join(filtered)

        size = cfg.chunk_size or 400
        try:
            chunks = split_into_chunks(joined, size)
        except Exception:  # pragma: no cover - fallback if chunking fails
            chunks = []
            if joined:
                chunks.append(type("C", (), {"text": joined})())

        # ``ContextBuilder`` is a heavy dependency so we import lazily and fall
        # back to a tiny stub when unavailable.  The builder is cached on the
        # instance to avoid repeated initialisation across calls.
        builder = getattr(self, "_summary_builder", None)
        if builder is None:
            try:  # pragma: no cover - best effort builder creation
                from context_builder_util import create_context_builder

                builder = create_context_builder()
            except Exception as exc:  # pragma: no cover - builder unavailable
                logger.warning("context builder unavailable: %s", exc)

                class _DummyBuilder:
                    def build(self, _: str) -> str:
                        return ""

                builder = _DummyBuilder()
            self._summary_builder = builder

        # Try to initialise a lightweight local LLM client.  This is optional
        # and failures are logged but otherwise ignored.
        llm = getattr(self, "_summary_llm", None)
        if llm is None:
            try:  # pragma: no cover - optional dependency
                from local_client import OllamaClient

                llm = OllamaClient()
            except Exception:
                try:  # pragma: no cover - secondary fallback
                    from local_client import VLLMClient

                    llm = VLLMClient()
                except Exception as exc:
                    logger.debug("no local LLM available: %s", exc)
                    llm = None
            self._summary_llm = llm

        summaries: List[str] = []
        chunk_hashes: List[str] = []
        for ch in chunks:
            digest = hashlib.sha256(ch.text.encode("utf-8")).hexdigest()
            chunk_hashes.append(digest)
            try:
                summary = summarize_snippet(ch.text, llm, context_builder=builder)
                if not summary:
                    raise ValueError("empty summary")
            except Exception as exc:  # pragma: no cover - summariser issues
                logger.exception("summary generation failed for %s", digest, exc_info=exc)
                summary = ch.text
            summary = generalise(summary, config=cfg, db_key=db_key)
            if summary:
                summaries.append(summary)

        self._last_chunk_meta = {
            "chunk_count": len(chunks),
            "chunk_hashes": chunk_hashes,
        }
        return " ".join(s for s in summaries if s)

    def _prepare_text_for_embedding(
        self,
        text: str,
        *,
        chunk_tokens: int | None = None,
        config: "PreprocessingConfig" | None = None,
        db_key: str | None = None,
    ) -> str:
        """Backward compatible wrapper for older callers.

        Older callers may still pass ``chunk_tokens`` to specify the desired
        chunk size.  Newer code should pass a :class:`PreprocessingConfig`
        instance to fully control splitting behaviour.
        """

        if config is None and chunk_tokens is not None:
            config = PreprocessingConfig(chunk_size=chunk_tokens)
        return self._split_and_summarise(text, config=config, db_key=db_key)

    def _extract_last_updated(self, record: Any) -> str | None:
        """Best-effort extraction of a last-updated timestamp from ``record``.

        Many database records expose their modification time under different
        keys.  This helper normalises a handful of common field names and
        returns an ISO formatted string when found.  Subclasses can override
        this method for custom behaviour.
        """

        if isinstance(record, dict):
            for key in (
                "last_updated",
                "last_modification_date",
                "updated_at",
                "updated",
                "modified_at",
                "modified",
            ):
                val = record.get(key)
                if not val:
                    continue
                if isinstance(val, datetime):
                    return val.isoformat()
                try:
                    return str(val)
                except Exception:  # pragma: no cover - defensive
                    return None
        return None

    # ------------------------------------------------------------------
    # methods expected to be overridden
    def vector(self, record: Any) -> List[float]:
        """Return an embedding vector for ``record``.

        Textual records are expected to be preprocessed via
        :meth:`_split_and_summarise` before being passed here. Subclasses can
        override this for non-textual records.
        """

        if isinstance(record, str):
            return self.encode_text(record)
        raise NotImplementedError

    def iter_records(self) -> Iterator[Tuple[Any, Any, str]]:
        """Yield ``(record_id, record, kind)`` tuples for backfilling.

        Override in subclasses that use :meth:`backfill_embeddings`.
        """

        raise NotImplementedError

    def license_text(self, record: Any) -> str | None:
        """Return textual content to scan for license violations.

        Subclasses can override this to extract text from structured
        records. By default, if ``record`` is a string it is returned as-is
        otherwise ``None`` is returned, skipping the license check.
        """

        return record if isinstance(record, str) else None

    # ------------------------------------------------------------------
    # index persistence
    def load_index(self) -> None:
        """Load the vector index and metadata from disk if available."""
        self._index_loaded = True
        if self.metadata_path.exists():
            data = json.loads(self.metadata_path.read_text())
            self._id_map = data.get("id_map", [])
            self._metadata = data.get("metadata", {})
            self._vector_dim = data.get("vector_dim", 0)
            if not self._id_map:
                self._id_map = list(self._metadata.keys())
        if self.backend == "annoy":
            if AnnoyIndex and self.index_path.exists() and self._vector_dim:
                self._index = AnnoyIndex(self._vector_dim, "angular")
                self._index.load(str(self.index_path))
            elif AnnoyIndex and self._metadata:
                self._rebuild_index()
        elif self.backend == "faiss":
            if faiss and self.index_path.exists() and self._vector_dim:
                self._index = faiss.read_index(str(self.index_path))
            elif faiss and self._metadata:
                self._rebuild_index()

    def _ensure_index_loaded(self) -> None:
        if not self._index_loaded:
            self.load_index()

    def save_index(self) -> None:
        """Persist vector index and metadata to disk."""
        if self._index is not None:
            if self.backend == "annoy":
                if AnnoyIndex:
                    self._index.save(str(self.index_path))
            elif self.backend == "faiss":
                if faiss:
                    faiss.write_index(self._index, str(self.index_path))
        data = {
            "id_map": self._id_map,
            "metadata": self._metadata,
            "vector_dim": self._vector_dim,
            "last_vectorization": datetime.utcnow().isoformat(),
        }
        self.metadata_path.write_text(json.dumps(data, indent=2))

    def _rebuild_index(self) -> None:
        """Rebuild vector index from stored metadata."""
        if not self._metadata:
            self._index = None
            return
        self._vector_dim = len(next(iter(self._metadata.values()))["vector"])
        if self.backend == "annoy":
            if not AnnoyIndex:
                self._index = None
                return
            self._index = AnnoyIndex(self._vector_dim, "angular")
            for i, rid in enumerate(self._id_map):
                vec = self._metadata[rid]["vector"]
                self._index.add_item(i, vec)
            self._index.build(10)
        elif self.backend == "faiss":
            if not faiss or not np:
                self._index = None
                return
            self._index = faiss.IndexFlatIP(self._vector_dim)
            vectors = [self._metadata[rid]["vector"] for rid in self._id_map]
            if vectors:
                arr = np.array(vectors, dtype="float32")
                self._index.add(arr)

    # ------------------------------------------------------------------
    # public API
    def add_embedding(
        self,
        record_id: Any,
        record: Any,
        kind: str,
        *,
        source_id: str = "",
        chunk_meta: Dict[str, Any] | None = None,
    ) -> None:
        """Embed ``record`` and store the vector and metadata."""
        last_updated = self._extract_last_updated(record)
        text = self.license_text(record)
        if text is None and isinstance(record, str):
            text = record
        if text:
            lic = license_check(text)
            if lic:
                try:  # pragma: no cover - best effort
                    hash_ = license_fingerprint(text)
                    log_fn = getattr(self, "log_license_violation", None)
                    if callable(log_fn):
                        log_fn("", lic, hash_)
                except Exception:  # pragma: no cover - best effort
                    logger.exception(
                        "failed to log license violation for %s", record_id
                    )
                rid = str(record_id)
                self._metadata[rid] = {
                    "created_at": datetime.utcnow().isoformat(),
                    "embedding_version": self.embedding_version,
                    "kind": kind,
                    "source_id": source_id,
                    "redacted": False,
                    "license": lic,
                }
                if last_updated:
                    self._metadata[rid]["last_updated"] = last_updated
                log_embedding_metrics(
                    self.__class__.__name__, 0, 0.0, 0.0, vector_id=str(record_id)
                )
                logger.warning(
                    "skipping embedding for %s due to license %s", record_id, lic
                )
                return
            alerts = find_semantic_risks(text.splitlines())
            if alerts:
                rid = str(record_id)
                self._metadata[rid] = {
                    "created_at": datetime.utcnow().isoformat(),
                    "embedding_version": self.embedding_version,
                    "kind": kind,
                    "source_id": source_id,
                    "redacted": False,
                    "semantic_risks": alerts,
                }
                if last_updated:
                    self._metadata[rid]["last_updated"] = last_updated
                log_embedding_metrics(
                    self.__class__.__name__, 0, 0.0, 0.0, vector_id=str(record_id)
                )
                logger.warning(
                    "skipping embedding for %s due to semantic risks", record_id
                )
                for line, msg, score in alerts:
                    logger.warning("semantic risk %.2f for %s: %s", score, line, msg)
                return
        record = redact(record) if isinstance(record, str) else record
        if isinstance(record, str):
            if chunk_meta is None:
                record = self._split_and_summarise(record)
                chunk_meta = getattr(self, "_last_chunk_meta", {})
        if chunk_meta is None:
            chunk_meta = {"chunk_count": 0, "chunk_hashes": []}

        start = perf_counter()
        vec = self.vector(record)
        chunk_meta = getattr(self, "_last_chunk_meta", chunk_meta)
        wall_time = perf_counter() - start
        tokens = getattr(self, "_last_embedding_tokens", 0)
        if not tokens and isinstance(record, str):  # pragma: no cover - best effort
            try:
                tokenizer = getattr(self.model, "tokenizer", None)
                if tokenizer:
                    tokens = len(tokenizer.encode(record))
            except Exception:  # pragma: no cover - best effort
                tokens = 0
        self._last_embedding_tokens = tokens
        self._last_embedding_time = wall_time

        rid = str(record_id)
        if rid not in self._metadata:
            self._id_map.append(rid)
        self._metadata[rid] = {
            "vector": list(vec),
            "created_at": datetime.utcnow().isoformat(),
            "embedding_version": self.embedding_version,
            "kind": kind,
            "source_id": source_id,
            "redacted": True,
            "record": record,
            **(chunk_meta or {}),
            "_last_chunk_meta": chunk_meta or {},
        }
        if last_updated:
            self._metadata[rid]["last_updated"] = last_updated
        self._rebuild_index()
        save_start = perf_counter()
        self.save_index()
        index_latency = perf_counter() - save_start
        log_embedding_metrics(
            self.__class__.__name__,
            tokens,
            wall_time,
            index_latency,
            vector_id=str(record_id),
        )

    def get_vector(self, record_id: Any) -> List[float] | None:
        """Return the stored embedding vector for ``record_id`` if present."""

        self._ensure_index_loaded()
        meta = self._metadata.get(str(record_id))
        if meta:
            self._record_staleness(str(record_id), meta.get("created_at"))
            return list(meta["vector"])
        return None

    def try_add_embedding(
        self,
        record_id: Any,
        record: Any,
        kind: str,
        *,
        source_id: str = "",
    ) -> None:
        """Best-effort variant of :meth:`add_embedding`."""

        try:
            self.add_embedding(record_id, record, kind, source_id=source_id)
        except Exception:  # pragma: no cover - best effort
            logging.exception("embedding hook failed for %s", record_id)

    def update_embedding_version(
        self, record_id: Any, *, embedding_version: int | None = None
    ) -> None:
        """Update ``embedding_version`` metadata for ``record_id``."""

        self._ensure_index_loaded()
        rid = str(record_id)
        if rid not in self._metadata:
            return
        version = (
            embedding_version if embedding_version is not None else self.embedding_version
        )
        self._metadata[rid]["embedding_version"] = int(version)
        self._metadata[rid]["created_at"] = datetime.utcnow().isoformat()
        self.save_index()

    def update_embedding_versions(
        self, record_ids: Sequence[Any], *, embedding_version: int | None = None
    ) -> None:
        """Bulk update ``embedding_version`` for multiple records."""

        self._ensure_index_loaded()
        version = (
            embedding_version if embedding_version is not None else self.embedding_version
        )
        now = datetime.utcnow().isoformat()
        updated = False
        for rid in map(str, record_ids):
            if rid in self._metadata:
                self._metadata[rid]["embedding_version"] = int(version)
                self._metadata[rid]["created_at"] = now
                updated = True
        if updated:
            self.save_index()

    def needs_refresh(self, record_id: Any, record: Any | None = None) -> bool:
        """Return ``True`` if ``record_id`` requires re-embedding.

        A record is considered stale when no metadata is stored, the
        ``embedding_version`` has changed, or the supplied ``record`` carries a
        different ``last_updated`` timestamp to that stored in metadata.  When
        ``record`` is ``None`` the check falls back to version mismatches only.
        """

        self._ensure_index_loaded()
        rid = str(record_id)
        meta = self._metadata.get(rid)
        if not meta:
            return True
        try:
            if int(meta.get("embedding_version", 0)) != int(self.embedding_version):
                return True
        except Exception:
            return True
        if record is not None:
            current = self._extract_last_updated(record)
            stored = meta.get("last_updated")
            if current and stored != current:
                return True
        return False

    # internal ---------------------------------------------------------
    def _record_staleness(self, rid: str, created_at: str | None) -> None:
        """Log how stale an embedding is when accessed."""
        if not created_at:
            return
        try:
            age = (datetime.utcnow() - datetime.fromisoformat(created_at)).total_seconds()
        except Exception:
            return
        origin = getattr(self, "origin_db", self.__class__.__name__)
        if _EMBED_STALE:
            try:
                _EMBED_STALE.labels(origin_db=origin).set(age)
            except ValueError:  # pragma: no cover - labels not configured
                _EMBED_STALE.set(age)
        try:
            _resolve_metrics_db()().log_embedding_staleness(origin, rid, age)
        except Exception:  # pragma: no cover - best effort
            logger.exception("failed to persist embedding staleness")

    def search_by_vector(
        self, vector: Sequence[float], top_k: int = 10
    ) -> List[Tuple[Any, float]]:
        """Return ``top_k`` records most similar to ``vector``."""

        self._ensure_index_loaded()
        if self._index is None:
            self.load_index()
        if self._index is None:
            return []
        if self.backend == "annoy":
            ids, dists = self._index.get_nns_by_vector(
                list(vector), top_k, include_distances=True
            )
            results: List[Tuple[Any, float]] = []
            for i, d in zip(ids, dists):
                if i < len(self._id_map):
                    rid = self._id_map[i]
                    meta = self._metadata.get(rid)
                    if not meta or not meta.get("redacted"):
                        continue
                    results.append((rid, float(d)))
                    self._record_staleness(rid, meta.get("created_at"))
            return results
        elif self.backend == "faiss":
            if not faiss or not np:
                return []
            vec = np.array([list(vector)], dtype="float32")
            dists, ids = self._index.search(vec, top_k)
            results: List[Tuple[Any, float]] = []
            for idx, dist in zip(ids[0], dists[0]):
                if 0 <= idx < len(self._id_map):
                    rid = self._id_map[idx]
                    meta = self._metadata.get(rid)
                    if not meta or not meta.get("redacted"):
                        continue
                    results.append((rid, float(dist)))
                    self._record_staleness(rid, meta.get("created_at"))
            return results
        return []

    def backfill_embeddings(self) -> None:
        """Generate embeddings for all records lacking them."""
        for record_id, record, kind in self.iter_records():
            rid = str(record_id)
            if not self.needs_refresh(record_id, record):
                continue
            text = self.license_text(record)
            if text is None and isinstance(record, str):
                text = record
            if text:
                lic = license_check(text)
                if lic:
                    hash_ = license_fingerprint(text)
                    log_fn = getattr(self, "log_license_violation", None)
                    if callable(log_fn):
                        try:  # pragma: no cover - best effort
                            log_fn("", lic, hash_)
                        except Exception:  # pragma: no cover - best effort
                            logger.exception(
                                "failed to log license violation for %s", record_id
                            )
                    self._metadata[rid] = {
                        "created_at": datetime.utcnow().isoformat(),
                        "embedding_version": self.embedding_version,
                        "kind": kind,
                        "source_id": "",
                        "redacted": False,
                        "license": lic,
                    }
                    last_updated = self._extract_last_updated(record)
                    if last_updated:
                        self._metadata[rid]["last_updated"] = last_updated
                    logger.warning(
                        "skipping embedding for %s due to license %s", record_id, lic
                    )
                    continue
                alerts = find_semantic_risks(text.splitlines())
                if alerts:
                    self._metadata[rid] = {
                        "created_at": datetime.utcnow().isoformat(),
                        "embedding_version": self.embedding_version,
                        "kind": kind,
                        "source_id": "",
                        "redacted": False,
                        "semantic_risks": alerts,
                    }
                    last_updated = self._extract_last_updated(record)
                    if last_updated:
                        self._metadata[rid]["last_updated"] = last_updated
                    logger.warning(
                        "skipping embedding for %s due to semantic risks", record_id
                    )
                    for line, msg, score in alerts:
                        logger.warning("semantic risk %.2f for %s: %s", score, line, msg)
                    continue
            record = redact(record) if isinstance(record, str) else record
            chunk_meta: Dict[str, Any] | None = None
            if isinstance(record, str):
                record = self._split_and_summarise(record)
                chunk_meta = getattr(self, "_last_chunk_meta", {})
            self.add_embedding(record_id, record, kind, chunk_meta=chunk_meta)


__all__ = ["EmbeddableDBMixin", "safe_super_init", "safe_super_init_or_warn"]
