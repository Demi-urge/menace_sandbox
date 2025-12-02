from __future__ import annotations

"""Shared vectorisation interface for disparate data sources.

This module consolidates the various standalone vectorisers into a single
service.  Callers provide a ``kind`` identifying the record type and a
dictionary representing the record.  The service delegates to the
appropriate vectoriser and optionally persists the resulting embedding
using a configurable :class:`~vector_service.vector_store.VectorStore`.
"""

# ruff: noqa: T201 - module level debug prints are routed via logging

from dataclasses import dataclass, field
import hashlib
import shutil
import threading
from datetime import datetime
from typing import Any, Callable, Dict, List

from pathlib import Path
import json
import logging
import os
import socket
import tarfile
import tempfile
import time
import urllib.error
import urllib.request

try:  # pragma: no cover - optional heavy dependency
    import torch
    from transformers import AutoModel, AutoTokenizer
except Exception:  # pragma: no cover - degrade gracefully when unavailable
    torch = None  # type: ignore[assignment]
    AutoModel = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]

try:  # pragma: no cover - package execution path
    from dynamic_path_router import resolve_path
except ModuleNotFoundError:  # pragma: no cover - fallback when run as ``python -m vectorizer``
    import sys

    _PACKAGE_ROOT = Path(__file__).resolve().parents[1]
    if str(_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_ROOT))
    from dynamic_path_router import resolve_path  # type: ignore

from governed_embeddings import governed_embed, get_embedder

try:  # pragma: no cover - prefer package-relative imports
    from .registry import (
        _VECTOR_REGISTRY,
        load_handler,
        load_handlers,
        _resolve_bootstrap_fast,
    )
    from .vector_store import VectorStore, get_default_vector_store
except ImportError as exc:  # pragma: no cover - fallback when executed as a script
    if "attempted relative import" not in str(exc):
        raise
    from vector_service.registry import (  # type: ignore
        _VECTOR_REGISTRY,
        load_handler,
        load_handlers,
        _resolve_bootstrap_fast,
    )
    from vector_service.vector_store import (  # type: ignore
        VectorStore,
        get_default_vector_store,
    )

try:  # pragma: no cover - optional dependency used for text embeddings
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - avoid hard dependency
    SentenceTransformer = None  # type: ignore

from .lazy_bootstrap import ensure_embedding_model


_BUNDLED_MODEL = resolve_path("vector_service/minilm") / "tiny-distilroberta-base.tar.xz"
_BUNDLED_MODEL_CACHE_ROOT = Path(tempfile.gettempdir()) / "vector_service" / "minilm"
_LOCAL_TOKENIZER: AutoTokenizer | None = None
_LOCAL_MODEL: AutoModel | None = None
_LOCAL_BUNDLE_CHECKSUM: str | None = None


logger = logging.getLogger(__name__)


def _trace(event: str, **extra: Any) -> None:
    """Emit lightweight diagnostics when ``VECTOR_SERVICE_TRACE`` is set."""

    flag = os.getenv("VECTOR_SERVICE_TRACE", "")
    if flag and flag.lower() not in {"0", "false", "no", "off"}:
        payload = {"event": event, **extra}
        logger.log(logging.INFO, "vector-service: %s", event, extra=payload)


def _timestamp_payload(start: float | None = None, **extra: Any) -> Dict[str, Any]:
    payload = {"ts": datetime.utcnow().isoformat(), **extra}
    if start is not None:
        payload["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 3)
    return payload


def _remote_timeout() -> float | None:
    raw = os.getenv("VECTOR_SERVICE_TIMEOUT", "10").strip()
    if not raw:
        return 10.0
    try:
        value = float(raw)
    except Exception:
        logger.warning(
            "invalid VECTOR_SERVICE_TIMEOUT=%r; defaulting to 10s", raw
        )
        return 10.0
    if value <= 0:
        logger.warning(
            "VECTOR_SERVICE_TIMEOUT must be positive; defaulting to 10s"
        )
        return 10.0
    return value


_REMOTE_URL = os.environ.get("VECTOR_SERVICE_URL")
_REMOTE_ENDPOINT = _REMOTE_URL.rstrip("/") if _REMOTE_URL else None
_REMOTE_TIMEOUT = _remote_timeout()
_REMOTE_DISABLED = False


def _load_local_model() -> tuple[AutoTokenizer, AutoModel]:
    """Load the bundled fallback embedding model."""

    global _LOCAL_TOKENIZER, _LOCAL_MODEL, _LOCAL_BUNDLE_CHECKSUM
    if AutoTokenizer is None or AutoModel is None or torch is None:
        raise RuntimeError("local embedding model dependencies unavailable")
    ensure_embedding_model(logger=logger)
    bundle_checksum = _compute_bundle_checksum()
    if bundle_checksum != _LOCAL_BUNDLE_CHECKSUM:
        _trace("local-model.bundle-changed")
        _LOCAL_TOKENIZER = None
        _LOCAL_MODEL = None
        _LOCAL_BUNDLE_CHECKSUM = bundle_checksum
        _cleanup_stale_bundle_caches(bundle_checksum)
    if _LOCAL_TOKENIZER is None or _LOCAL_MODEL is None:
        cache_dir = _ensure_cached_model(bundle_checksum)
        _LOCAL_TOKENIZER = AutoTokenizer.from_pretrained(cache_dir)
        _LOCAL_MODEL = AutoModel.from_pretrained(cache_dir)
        _LOCAL_MODEL.eval()
    return _LOCAL_TOKENIZER, _LOCAL_MODEL


def _compute_bundle_checksum() -> str:
    if not _BUNDLED_MODEL.exists():
        raise FileNotFoundError(
            f"bundled model archive missing at {_BUNDLED_MODEL} "
            "- run `python -m vector_service.download_model` to fetch it"
        )
    hasher = hashlib.sha256()
    with open(_BUNDLED_MODEL, "rb") as bundle:
        for chunk in iter(lambda: bundle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _cleanup_stale_bundle_caches(active_checksum: str) -> None:
    if not _BUNDLED_MODEL_CACHE_ROOT.exists():
        return
    for entry in _BUNDLED_MODEL_CACHE_ROOT.iterdir():
        if entry.name != active_checksum and entry.is_dir():
            shutil.rmtree(entry, ignore_errors=True)


def _ensure_cached_model(checksum: str) -> Path:
    target_dir = _BUNDLED_MODEL_CACHE_ROOT / checksum
    if _is_cache_intact(target_dir):
        _trace("local-model.cache-hit", path=str(target_dir))
        return target_dir
    if target_dir.exists():
        shutil.rmtree(target_dir, ignore_errors=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(_BUNDLED_MODEL) as tar:
        tar.extractall(target_dir)
    return target_dir


def _is_cache_intact(target_dir: Path) -> bool:
    required_files = ["config.json", "tokenizer.json"]
    has_weights = any(target_dir.glob("*.bin"))
    return target_dir.exists() and all(
        (target_dir / file_name).exists() for file_name in required_files
    ) and has_weights


def _local_embed(text: str) -> List[float]:
    """Return an embedding using the bundled model."""

    if torch is None:
        raise RuntimeError("local embedding model dependencies unavailable")
    tokenizer, model = _load_local_model()
    inputs = tokenizer([text], padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs)
    embedding = output.last_hidden_state.mean(dim=1)[0]
    return [float(x) for x in embedding.tolist()]


@dataclass
class SharedVectorService:
    """Facade exposing a unified ``vectorise`` API."""

    text_embedder: SentenceTransformer | None = None
    vector_store: VectorStore | None = None
    bootstrap_fast: bool | None = None
    hydrate_handlers: bool | None = None
    lazy_vector_store: bool | None = None
    warmup_lite: bool | None = None
    stop_event: threading.Event | None = None
    budget_check: Callable[[threading.Event | None], None] | None = None
    _handlers: Dict[str, Callable[[Dict[str, Any]], List[float]]] = field(init=False)
    _handler_requires_store: Dict[str, bool] = field(init=False)
    _handler_bootstrap_flag: bool = field(init=False, default=False)
    _known_kinds: set[str] = field(init=False, default_factory=set)
    _handler_hydration_deferred: bool = field(init=False, default=False)
    _handler_deferral_reasons: set[str] = field(init=False, default_factory=set)
    _background_hydration_started: bool = field(init=False, default=False)
    _background_hydration_thread: threading.Thread | None = field(
        init=False, default=None
    )

    def __post_init__(self) -> None:
        self._check_cancelled("init")
        # Handlers are populated dynamically from the registry so newly
        # registered vectorisers are picked up automatically.
        init_start = time.perf_counter()
        logger.info(
            "shared_vector_service.init.start",
            extra=_timestamp_payload(init_start),
        )
        handler_start = time.perf_counter()
        requested_fast = self.bootstrap_fast
        resolved_fast, bootstrap_context, defaulted_fast = _resolve_bootstrap_fast(
            requested_fast
        )
        warmup_lite = bool(self.warmup_lite)
        self.warmup_lite = warmup_lite
        if self.bootstrap_fast is None:
            self.bootstrap_fast = resolved_fast
        if self.hydrate_handlers is None:
            self.hydrate_handlers = False
        if self.lazy_vector_store is None:
            self.lazy_vector_store = bootstrap_context or resolved_fast
        if warmup_lite:
            self.lazy_vector_store = True
        self._handler_bootstrap_flag = resolved_fast
        self._handlers = {}
        self._handler_requires_store = {}
        self._handler_lock = threading.RLock()
        self._vector_store_lock = threading.RLock()
        self._known_kinds = set()
        if bootstrap_context and resolved_fast:
            logger.info(
                "shared_vector_service.bootstrap_fast.active",
                extra={
                    "bootstrap_fast_defaulted": defaulted_fast,
                    "bootstrap_context": True,
                },
            )
        self._known_kinds = set(_VECTOR_REGISTRY.keys())
        if self.hydrate_handlers:
            self.hydrate_all_handlers(stop_event=self.stop_event)
        else:
            self._mark_handler_deferral(
                "deferred-init" if not warmup_lite else "warmup-lite",
                schedule_background=True,
            )
            logger.info(
                "shared_vector_service.handlers.deferred",
                extra=_timestamp_payload(
                    handler_start,
                    handler_count=len(self._known_kinds),
                    bootstrap_fast_active=resolved_fast,
                    bootstrap_context=bootstrap_context,
                    warmup_lite=warmup_lite,
                ),
            )
        if self.vector_store is not None:
            self.lazy_vector_store = False
        if warmup_lite:
            logger.info(
                "shared_vector_service.warmup_lite.active",
                extra=_timestamp_payload(
                    handler_start,
                    bootstrap_fast_active=resolved_fast,
                    bootstrap_context=bootstrap_context,
                ),
            )
        _trace(
            "shared_vector_service.init.complete",
            has_vector_store=self.vector_store is not None,
        )
        logger.info(
            "shared_vector_service.init.complete",
            extra=_timestamp_payload(
                init_start,
                has_vector_store=self.vector_store is not None,
                handler_count=len(self._handlers),
            ),
        )

    @property
    def handler_hydration_deferred(self) -> bool:
        return self._handler_hydration_deferred

    @property
    def handler_deferral_reasons(self) -> set[str]:
        return set(self._handler_deferral_reasons)

    def schedule_background_hydration(self) -> None:
        """Expose a best-effort background hydration trigger."""

        self._schedule_background_hydration()

    def hydrate_all_handlers(
        self,
        *,
        stop_event: threading.Event | None = None,
        timeout: float | None = None,
        budget_check: Callable[[threading.Event | None], None] | None = None,
    ) -> None:
        """Instantiate and cache all registered handlers."""

        handler_start = time.perf_counter()
        try:
            self._check_budget_deadline(
                "hydrate-handlers", handler_start, timeout, stop_event, budget_check
            )
            handlers = load_handlers(
                bootstrap_fast=self._handler_bootstrap_flag,
                warmup_lite=bool(self.warmup_lite),
                stop_event=stop_event or self.stop_event,
                budget_check=budget_check or self.budget_check,
            )
            self._check_budget_deadline(
                "hydrate-handlers", handler_start, timeout, stop_event, budget_check
            )
        except TimeoutError:
            self._mark_handler_deferral("timeout", schedule_background=True)
            logger.info(
                "shared_vector_service.handlers.deferred", extra={"reason": "timeout"}
            )
            return

        self._clear_handler_deferral()
        self._handlers.update(handlers)
        for kind, handler in handlers.items():
            self._handler_requires_store[kind] = self._handler_requires_store.get(
                kind, self._handler_requires_store_flag(handler)
            )
        logger.info(
            "shared_vector_service.handlers.loaded",
            extra=_timestamp_payload(
                handler_start,
                handler_count=len(self._handlers),
                bootstrap_fast_active=self._handler_bootstrap_flag,
            ),
        )
        patch_handler = self._handlers.get("patch")
        patch_handler_deferred = getattr(patch_handler, "is_patch_stub", False)
        if patch_handler_deferred:
            logger.info(
                "shared_vector_service.bootstrap_fast.patch_handler_deferred",
                extra=_timestamp_payload(
                    handler_start,
                    handler_count=len(self._handlers),
                    bootstrap_fast_active=True,
                    bootstrap_fast_defaulted=self.bootstrap_fast is None,
                    deferred_patch=True,
                ),
            )
        _trace(
            "shared_vector_service.handlers.loaded",
            handler_count=len(self._handlers),
            handlers=sorted(self._handlers.keys()),
        )

    def _should_skip_vector_store(self) -> bool:
        return bool(self.bootstrap_fast) or bool(self.warmup_lite)

    def _initialise_vector_store(
        self,
        *,
        force: bool = False,
        stop_event: threading.Event | None = None,
        timeout: float | None = None,
        budget_check: Callable[[threading.Event | None], None] | None = None,
    ) -> None:
        event = stop_event or self.stop_event
        budget_hook = budget_check or self.budget_check
        if self.vector_store is not None:
            return
        if not force and (self.lazy_vector_store or self._should_skip_vector_store()):
            logger.info(
                "shared_vector_service.vector_store.deferred",
                extra={"lazy": bool(self.lazy_vector_store), "bootstrap_fast": bool(self.bootstrap_fast)},
            )
            return
        start = time.perf_counter()
        try:
            self._check_budget_deadline(
                "vector-store", start, timeout, event, budget_hook
            )
        except TimeoutError:
            logger.info(
                "shared_vector_service.vector_store.deferred", extra={"reason": "timeout"}
            )
            return
        with self._vector_store_lock:
            if self.vector_store is not None:
                return
            _trace("shared_vector_service.vector_store.fetch")
            try:
                self._check_budget_deadline(
                    "vector-store", start, timeout, event, budget_hook
                )
            except TimeoutError:
                logger.info(
                    "shared_vector_service.vector_store.deferred",
                    extra={"reason": "timeout-lock"},
                )
                return
            store_start = time.perf_counter()
            try:
                self.vector_store = get_default_vector_store(
                    lazy=bool(self.lazy_vector_store)
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                _trace("shared_vector_service.vector_store.error", error=str(exc))
                raise
            try:
                self._check_budget_deadline(
                    "vector-store", start, timeout, event, budget_hook
                )
            except TimeoutError:
                logger.info(
                    "shared_vector_service.vector_store.partial", extra={"reason": "timeout"}
                )
                return
            logger.info(
                "shared_vector_service.vector_store.resolved",
                extra=_timestamp_payload(
                    store_start,
                    resolved=bool(self.vector_store),
                    lazy=bool(self.lazy_vector_store),
                ),
            )

    def _handler_requires_store_flag(
        self, handler: Callable[[Dict[str, Any]], List[float]]
    ) -> bool:
        for attr in ("requires_vector_store", "needs_vector_store", "uses_vector_store"):
            value = getattr(handler, attr, None)
            if isinstance(value, bool):
                return value
            if value is not None:
                return bool(value)
        bound = getattr(handler, "__self__", None)
        if bound is None:
            return False
        for attr in ("requires_vector_store", "needs_vector_store", "uses_vector_store"):
            value = getattr(bound, attr, None)
            if isinstance(value, bool):
                return value
            if value is not None:
                return bool(value)
        return False

    def _check_cancelled(
        self,
        context: str,
        stop_event: threading.Event | None = None,
        budget_check: Callable[[threading.Event | None], None] | None = None,
    ) -> None:
        event = stop_event or self.stop_event
        if event is not None and event.is_set():
            raise TimeoutError(f"shared vector service cancelled during {context}")
        hook = budget_check or self.budget_check
        if hook is not None:
            hook(event)

    def _check_budget_deadline(
        self,
        context: str,
        start: float,
        timeout: float | None,
        stop_event: threading.Event | None,
        budget_check: Callable[[threading.Event | None], None] | None,
    ) -> None:
        self._check_cancelled(context, stop_event, budget_check)
        if timeout is None:
            return
        if (time.perf_counter() - start) >= timeout:
            raise TimeoutError(f"shared vector service {context} timed out")

    def _mark_handler_deferral(
        self, reason: str, *, schedule_background: bool = False
    ) -> None:
        self._handler_hydration_deferred = True
        self._handler_deferral_reasons.add(reason)
        if schedule_background:
            self._schedule_background_hydration()

    def _clear_handler_deferral(self) -> None:
        self._handler_hydration_deferred = False
        self._handler_deferral_reasons.clear()

    def _schedule_background_hydration(self) -> None:
        if self._background_hydration_started:
            return
        if self.stop_event is not None and self.stop_event.is_set():
            return
        self._background_hydration_started = True

        def _runner() -> None:
            try:
                self.hydrate_all_handlers(stop_event=self.stop_event)
            except Exception:  # pragma: no cover - best effort background hydration
                logger.debug(
                    "background handler hydration failed", exc_info=True
                )

        self._background_hydration_thread = threading.Thread(
            target=_runner,
            name="shared-vector-service-handler-hydration",
            daemon=True,
        )
        self._background_hydration_thread.start()

    def _prepare_vector_store_for_handler(
        self,
        kind: str,
        handler: Callable[[Dict[str, Any]], List[float]],
        *,
        stop_event: threading.Event | None = None,
        budget_check: Callable[[threading.Event | None], None] | None = None,
    ) -> None:
        self._check_cancelled("prepare-vector-store", stop_event, budget_check)
        requires_store = self._handler_requires_store.get(kind)
        if requires_store is None:
            requires_store = self._handler_requires_store_flag(handler)
            self._handler_requires_store[kind] = requires_store
        if not requires_store:
            return
        if self.vector_store is None and self._should_skip_vector_store():
            logger.info(
                "shared_vector_service.vector_store.skipped",
                extra={"kind": kind, "bootstrap_fast": bool(self.bootstrap_fast)},
            )
            return
        self._initialise_vector_store(
            force=True,
            stop_event=stop_event or self.stop_event,
            budget_check=budget_check or self.budget_check,
        )

    def _get_handler(self, kind: str) -> Callable[[Dict[str, Any]], List[float]] | None:
        normalised = kind.lower()
        handler = self._handlers.get(normalised)
        if handler is not None:
            return handler
        if normalised not in self._known_kinds:
            return None
        with self._handler_lock:
            handler = self._handlers.get(normalised)
            if handler is not None:
                return handler
            handler_start = time.perf_counter()
            handler = load_handler(normalised, bootstrap_fast=self._handler_bootstrap_flag)
            if handler is not None:
                self._handlers[normalised] = handler
                self._handler_requires_store[normalised] = self._handler_requires_store_flag(
                    handler
                )
                logger.info(
                    "shared_vector_service.handler.lazy_loaded",
                    extra=_timestamp_payload(
                        handler_start,
                        kind=normalised,
                        bootstrap_fast=self._handler_bootstrap_flag,
                    ),
                )
            else:
                logger.info(
                    "shared_vector_service.handler.unavailable",
                    extra=_timestamp_payload(handler_start, kind=normalised),
                )
        return handler

    def _ensure_text_embedder(self) -> SentenceTransformer | None:
        """Initialise ``self.text_embedder`` if possible."""

        if self.text_embedder is not None:
            return self.text_embedder
        if SentenceTransformer is None:
            return None
        _trace("shared_vector_service.embedder.resolve.start")
        embedder = get_embedder()
        if embedder is None:
            _trace("shared_vector_service.embedder.resolve.failed")
            return None
        self.text_embedder = embedder
        _trace("shared_vector_service.embedder.resolve.success")
        return embedder

    def _encode_text(self, text: str) -> List[float]:
        embedder = self._ensure_text_embedder()
        embedder_available = embedder is not None
        if embedder_available or SentenceTransformer is not None:
            _trace(
                "shared_vector_service.encode_text.embedder",
                source="governed",
                embedder_available=embedder_available,
            )
            vec = governed_embed(text, embedder)
            if vec is not None:
                return [float(x) for x in vec]
            if embedder_available:
                raise RuntimeError("embedding failed")
        if SentenceTransformer is None:
            # SentenceTransformer not installed: load bundled model
            _trace("shared_vector_service.encode_text.embedder", source="local_fallback")
            return _local_embed(text)
        raise RuntimeError("text embedder unavailable")

    def _call_remote(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any] | None:
        global _REMOTE_DISABLED
        if _REMOTE_DISABLED or _REMOTE_ENDPOINT is None:
            return None
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{_REMOTE_ENDPOINT}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(
                req, timeout=_REMOTE_TIMEOUT or None
            ) as resp:  # pragma: no cover - network
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            logger.warning(
                "remote vector service unavailable; falling back to local handling",  # pragma: no cover - diagnostics
                extra={"endpoint": _REMOTE_ENDPOINT, "path": path, "error": str(exc)},
            )
        except json.JSONDecodeError as exc:
            logger.warning(
                "remote vector service returned invalid JSON; falling back to local handling",  # pragma: no cover - diagnostics
                extra={"endpoint": _REMOTE_ENDPOINT, "path": path, "error": str(exc)},
            )
        _REMOTE_DISABLED = True
        return None

    def ready(self) -> bool:
        """Return ``True`` when the service can respond to a request."""

        if _REMOTE_ENDPOINT is not None and not _REMOTE_DISABLED:
            return True
        if self._handlers:
            return True
        if self.warmup_lite:
            return bool(self._known_kinds or SentenceTransformer is not None or torch is not None)

        probe_kinds = list(self._known_kinds)[:3]
        for kind in probe_kinds:
            try:
                handler = self._get_handler(kind)
            except Exception:
                continue
            if handler is not None:
                return True
        return bool(SentenceTransformer is not None or torch is not None)

    def vectorise(
        self, kind: str, record: Dict[str, Any], *, stop_event: threading.Event | None = None
    ) -> List[float]:
        """Return an embedding for ``record`` of type ``kind``."""
        event = stop_event or self.stop_event
        self._check_cancelled("vectorise", event)
        _trace("shared_vector_service.vectorise.start", kind=kind)
        payload = self._call_remote("/vectorise", {"kind": kind, "record": record})
        if payload is not None:
            vec = payload.get("vector", [])
            if isinstance(vec, list):
                _trace("shared_vector_service.vectorise.remote", kind=kind)
                return vec

        self._check_cancelled("vectorise", event)

        kind = kind.lower()
        handler = self._get_handler(kind)
        if handler:
            self._check_cancelled("vectorise", event)
            self._prepare_vector_store_for_handler(
                kind, handler, stop_event=event, budget_check=self.budget_check
            )
            _trace("shared_vector_service.vectorise.handler", kind=kind)
            return handler(record)
        if kind in {"text", "prompt"}:
            _trace("shared_vector_service.vectorise.text", kind=kind)
            return self._encode_text(str(record.get("text", "")))
        if kind in {"stack"}:
            _trace("shared_vector_service.vectorise.stack", kind=kind)
            return self._encode_text(str(record.get("text", "")))
        raise ValueError(f"unknown record type: {kind}")

    def vectorise_and_store(
        self,
        kind: str,
        record_id: str,
        record: Dict[str, Any],
        *,
        origin_db: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> List[float]:
        """Vectorise ``record`` and persist the embedding."""

        vec = self.vectorise(kind, record)
        payload = self._call_remote(
            "/vectorise-and-store",
            {
                "kind": kind,
                "record_id": record_id,
                "record": record,
                "origin_db": origin_db,
                "metadata": metadata,
            },
        )
        if payload is not None:
            vec = payload.get("vector", vec)

        if self._should_skip_vector_store():
            logger.info(
                "shared_vector_service.vector_store.skipped", extra={"kind": kind, "reason": "bootstrap"}
            )
            return vec
        if self.vector_store is None:
            self._initialise_vector_store(
                force=True, stop_event=self.stop_event, budget_check=self.budget_check
            )
        if self.vector_store is None:
            raise RuntimeError("VectorStore not configured")
        self.vector_store.add(
            kind.lower(),
            record_id,
            vec,
            origin_db=origin_db or kind,
            metadata=metadata or {},
        )
        return vec


def update_workflow_embeddings(db_path: str = "workflows.db") -> None:
    """Embed all workflows in ``db_path`` using :class:`SharedVectorService`."""

    try:  # pragma: no cover - optional dependency
        from dataclasses import asdict
        from task_handoff_bot import WorkflowDB  # type: ignore
    except Exception:  # pragma: no cover - best effort
        return

    svc = SharedVectorService()
    db = WorkflowDB(Path(db_path))
    for wid, rec, _ in db.iter_records():
        try:
            svc.vectorise_and_store(
                "workflow",
                str(wid),
                asdict(rec),
                origin_db="workflow",
            )
        except Exception:  # pragma: no cover - best effort
            continue

__all__ = ["SharedVectorService", "update_workflow_embeddings"]
