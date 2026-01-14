from __future__ import annotations

"""Shared vectorisation interface for disparate data sources.

This module consolidates the various standalone vectorisers into a single
service.  Callers provide a ``kind`` identifying the record type and a
dictionary representing the record.  The service delegates to the
appropriate vectoriser and optionally persists the resulting embedding
using a configurable :class:`~vector_service.vector_store.VectorStore`.
"""

# ruff: noqa: T201 - module level debug prints are routed via logging

from concurrent.futures import Future, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
import hashlib
import random
import shutil
import threading
from datetime import datetime
from typing import Any, Callable, Dict, List, Mapping

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
import unicodedata

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

import governed_embeddings
from governed_embeddings import governed_embed, get_embedder
import metrics_exporter as _metrics
from redaction_utils import redact_text

try:  # pragma: no cover - prefer package-relative imports
    from .registry import (
        _VECTOR_REGISTRY,
        HandlerLoadResult,
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
        HandlerLoadResult,
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

from .lazy_bootstrap import (
    _BACKGROUND_QUEUE_FLAG,
    _HANDLER_VECTOR_MIN_BUDGET,
    _HEAVY_STAGE_CEILING,
    _update_warmup_stage_cache,
    ensure_embedding_model,
    ensure_embedding_model_future,
)


_BUNDLED_MODEL = resolve_path("vector_service/minilm") / "tiny-distilroberta-base.tar.xz"
_BUNDLED_MODEL_CACHE_ROOT = Path(tempfile.gettempdir()) / "vector_service" / "minilm"
_LOCAL_TOKENIZER: AutoTokenizer | None = None
_LOCAL_MODEL: AutoModel | None = None
_LOCAL_BUNDLE_CHECKSUM: str | None = None
_LOCAL_MODEL_PREP_FUTURE: Future | None = None
_LOCAL_MODEL_PREP_LOCK = threading.Lock()


VECTOR_EMBEDDER_RESOLVE_TOTAL = getattr(
    _metrics,
    "vector_embedder_resolve_total",
    _metrics.Gauge(
        "vector_embedder_resolve_total",
        "Vector embedder resolution outcomes",  # type: ignore[arg-type]
        ["status"],
    ),
)


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


def _normalize_redacted_text(text: str) -> str:
    redacted = redact_text(text)
    normalized = unicodedata.normalize("NFKC", redacted)
    normalized = normalized.replace("\x00", "")
    return normalized.strip()


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


def _remote_attempts() -> int:
    raw = os.getenv("VECTOR_SERVICE_REMOTE_ATTEMPTS", "6").strip()
    if not raw:
        return 6
    try:
        value = int(raw)
    except Exception:
        logger.warning(
            "invalid VECTOR_SERVICE_REMOTE_ATTEMPTS=%r; defaulting to 6",
            raw,
        )
        return 6
    if value <= 0:
        logger.warning(
            "VECTOR_SERVICE_REMOTE_ATTEMPTS must be positive; defaulting to 6",
        )
        return 6
    return value


def _remote_retry_delay() -> float:
    raw = os.getenv("VECTOR_SERVICE_REMOTE_RETRY_DELAY", "0.25").strip()
    if not raw:
        return 0.25
    try:
        value = float(raw)
    except Exception:
        logger.warning(
            "invalid VECTOR_SERVICE_REMOTE_RETRY_DELAY=%r; defaulting to 0.25",
            raw,
        )
        return 0.25
    if value <= 0:
        logger.warning(
            "VECTOR_SERVICE_REMOTE_RETRY_DELAY must be positive; defaulting to 0.25",
        )
        return 0.25
    return value


def _embedder_ceiling() -> float | None:
    raw = os.getenv("VECTOR_SERVICE_EMBEDDER_CEILING", "").strip()
    if raw == "":
        return _HEAVY_STAGE_CEILING
    try:
        value = float(raw)
    except Exception:
        logger.warning(
            "invalid VECTOR_SERVICE_EMBEDDER_CEILING=%r; using heavy stage ceiling %s",
            raw,
            _HEAVY_STAGE_CEILING,
        )
        return _HEAVY_STAGE_CEILING
    if value <= 0:
        logger.warning(
            "VECTOR_SERVICE_EMBEDDER_CEILING must be positive; disabling ceiling",
        )
        return None
    return value


_REMOTE_URL = os.environ.get("VECTOR_SERVICE_URL")
_REMOTE_ENDPOINT = _REMOTE_URL.rstrip("/") if _REMOTE_URL else None
_REMOTE_TIMEOUT = _remote_timeout()
_REMOTE_ATTEMPTS = _remote_attempts()
_REMOTE_RETRY_DELAY = _remote_retry_delay()
_REMOTE_DISABLED = False
_EMBEDDER_WAIT_CEILING = _embedder_ceiling()
_VECTOR_SERVICE_BOOT_TS = time.monotonic()
_REMOTE_BOOT_SKIP_LOGGED = False


def _remote_boot_skip_secs() -> float:
    raw = os.getenv("VECTOR_SERVICE_REMOTE_BOOT_SKIP_SECS", "0").strip()
    if raw == "":
        return 0.0
    try:
        value = float(raw)
    except Exception:
        logger.warning(
            "invalid VECTOR_SERVICE_REMOTE_BOOT_SKIP_SECS=%r; disabling boot skip",
            raw,
        )
        return 0.0
    if value <= 0:
        return 0.0
    return value


def _remote_ready_retry_config() -> tuple[int, float, float]:
    raw_retries = os.getenv("VECTOR_SERVICE_READY_RETRIES", "6").strip()
    raw_delay = os.getenv("VECTOR_SERVICE_READY_DELAY", "0.25").strip()
    raw_backoff = os.getenv("VECTOR_SERVICE_READY_BACKOFF", "1.5").strip()
    try:
        retries = max(1, int(raw_retries))
    except Exception:
        logger.warning(
            "invalid VECTOR_SERVICE_READY_RETRIES=%r; defaulting to 6",
            raw_retries,
        )
        retries = 6
    try:
        delay = max(0.05, float(raw_delay))
    except Exception:
        logger.warning(
            "invalid VECTOR_SERVICE_READY_DELAY=%r; defaulting to 0.25",
            raw_delay,
        )
        delay = 0.25
    try:
        backoff = max(1.0, float(raw_backoff))
    except Exception:
        logger.warning(
            "invalid VECTOR_SERVICE_READY_BACKOFF=%r; defaulting to 1.5",
            raw_backoff,
        )
        backoff = 1.5
    return retries, delay, backoff


def _probe_remote_ready() -> bool:
    if _REMOTE_ENDPOINT is None:
        return False
    ready_url = f"{_REMOTE_ENDPOINT}/health/ready"
    timeout = min(_REMOTE_TIMEOUT or 2.0, 2.0)
    try:
        with urllib.request.urlopen(ready_url, timeout=timeout):
            return True
    except Exception as exc:
        logger.debug("remote vector service readiness probe failed: %s", exc)
        return False


def _wait_for_remote_ready() -> bool:
    retries, delay, backoff = _remote_ready_retry_config()
    wait = delay
    for _ in range(retries):
        if _probe_remote_ready():
            return True
        time.sleep(wait)
        wait *= backoff
    return False


def _wait_for_remote_ready_short() -> bool:
    attempts = 5
    for attempt in range(attempts):
        if _probe_remote_ready():
            return True
        if attempt < attempts - 1:
            time.sleep(random.uniform(2.0, 5.0))
    return False


def _wait_for_remote_ready_cold_start() -> bool:
    attempts = 3
    delay = 2.0
    for attempt in range(attempts):
        if _probe_remote_ready():
            return True
        if attempt < attempts - 1:
            time.sleep(delay)
    return False


def _load_local_model() -> tuple[AutoTokenizer, AutoModel]:
    """Load the bundled fallback embedding model."""

    global _LOCAL_TOKENIZER, _LOCAL_MODEL, _LOCAL_BUNDLE_CHECKSUM
    if AutoTokenizer is None or AutoModel is None or torch is None:
        raise RuntimeError("local embedding model dependencies unavailable")
    stage_ceiling = _HEAVY_STAGE_CEILING
    wait_timeout = _REMOTE_TIMEOUT
    if stage_ceiling is not None:
        wait_timeout = min(stage_ceiling, wait_timeout) if wait_timeout else stage_ceiling
    warmup_thread = _is_warmup_thread()
    bundle_present = _BUNDLED_MODEL.exists()
    ceiling_known = stage_ceiling is not None and wait_timeout is not None
    if warmup_thread and (not bundle_present or not ceiling_known):
        background_timeout = wait_timeout if wait_timeout is not None else stage_ceiling
        _update_warmup_stage_cache(
            "model",
            "deferred-ceiling",
            logger,
            meta={"background_state": "queued", "background_timeout": background_timeout},
        )
        deferred = TimeoutError("local embedding model download deferred")
        setattr(deferred, "_deferred_status", _BACKGROUND_QUEUE_FLAG)
        setattr(deferred, "_deferred_timeout", background_timeout)
        raise deferred
    model_future = ensure_embedding_model_future(
        logger=logger,
        warmup=True,
        warmup_lite=False,
        warmup_heavy=True,
        download_timeout=_REMOTE_TIMEOUT,
    )
    prep_future = _get_local_model_prep_future(model_future)
    if warmup_thread and (not bundle_present or not ceiling_known):
        _update_warmup_stage_cache(
            "model",
            "deferred-ceiling",
            logger,
            meta={"background_state": "queued", "background_timeout": wait_timeout},
        )
        deferred = TimeoutError("local embedding model download deferred")
        setattr(deferred, "_deferred_status", _BACKGROUND_QUEUE_FLAG)
        setattr(deferred, "_deferred_timeout", wait_timeout)
        raise deferred
    if warmup_thread and not prep_future.done():
        _update_warmup_stage_cache(
            "model",
            "deferred-ceiling",
            logger,
            meta={"background_state": "queued", "background_timeout": wait_timeout},
        )
        deferred = TimeoutError("local embedding model download deferred")
        setattr(deferred, "_deferred_status", _BACKGROUND_QUEUE_FLAG)
        setattr(deferred, "_deferred_timeout", wait_timeout)
        raise deferred
    should_block = not warmup_thread and (wait_timeout is None or wait_timeout > 0)
    try:
        cache_dir = prep_future.result(timeout=wait_timeout) if should_block else prep_future.result(0)
    except FutureTimeout as exc:
        _trace(
            "local-model.deferred",
            timeout=wait_timeout,
            stage_ceiling=stage_ceiling,
        )
        deferred = TimeoutError("local embedding model download deferred")
        setattr(deferred, "_deferred_status", _BACKGROUND_QUEUE_FLAG)
        setattr(deferred, "_deferred_timeout", wait_timeout)
        raise deferred from exc
    except TimeoutError as exc:
        if getattr(exc, "_deferred_status", None) == _BACKGROUND_QUEUE_FLAG:
            raise
        raise
    if _LOCAL_TOKENIZER is None or _LOCAL_MODEL is None:
        _LOCAL_TOKENIZER = AutoTokenizer.from_pretrained(cache_dir)
        _LOCAL_MODEL = AutoModel.from_pretrained(cache_dir)
        _LOCAL_MODEL.eval()
    return _LOCAL_TOKENIZER, _LOCAL_MODEL


def _is_warmup_thread() -> bool:
    name = threading.current_thread().name.lower()
    return "warmup" in name or "bootstrap" in name


def _get_local_model_prep_future(model_future: Future) -> Future:
    global _LOCAL_MODEL_PREP_FUTURE, _LOCAL_BUNDLE_CHECKSUM, _LOCAL_TOKENIZER, _LOCAL_MODEL
    with _LOCAL_MODEL_PREP_LOCK:
        if _LOCAL_MODEL_PREP_FUTURE is not None and not _LOCAL_MODEL_PREP_FUTURE.cancelled():
            if _LOCAL_MODEL_PREP_FUTURE.done() and _LOCAL_MODEL_PREP_FUTURE.exception() is not None:
                _LOCAL_MODEL_PREP_FUTURE = None
            else:
                return _LOCAL_MODEL_PREP_FUTURE

        prep_future: Future = Future()

        def _prepare() -> None:
            try:
                result = model_future.result()
                bundle_path = result[0] if isinstance(result, tuple) else result
                if bundle_path is None:
                    deferred = TimeoutError("local embedding model download deferred")
                    setattr(deferred, "_deferred_status", _BACKGROUND_QUEUE_FLAG)
                    setattr(deferred, "_deferred_timeout", None)
                    raise deferred
                bundle_checksum = _compute_bundle_checksum()
                if bundle_checksum != _LOCAL_BUNDLE_CHECKSUM:
                    _trace("local-model.bundle-changed")
                    _LOCAL_TOKENIZER = None
                    _LOCAL_MODEL = None
                    _LOCAL_BUNDLE_CHECKSUM = bundle_checksum
                    _cleanup_stale_bundle_caches(bundle_checksum)
                cache_dir = _ensure_cached_model(bundle_checksum)
            except Exception as exc:  # pragma: no cover - background preparation
                prep_future.set_exception(exc)
            else:
                prep_future.set_result(cache_dir)

        thread = threading.Thread(
            target=_prepare, name="local-embedder-cache", daemon=True
        )
        thread.start()
        _LOCAL_MODEL_PREP_FUTURE = prep_future
        return prep_future


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
    handler_timeouts: Mapping[str, float] | float | None = None
    _handlers: Dict[str, Callable[[Dict[str, Any]], List[float]]] = field(init=False)
    _handler_requires_store: Dict[str, bool] = field(init=False)
    _handler_bootstrap_flag: bool = field(init=False, default=False)
    _known_kinds: set[str] = field(init=False, default_factory=set)
    _handlers_ready_flag: bool = field(init=False, default=False)
    _handler_hydration_deferred: bool = field(init=False, default=False)
    _handler_deferral_reasons: set[str] = field(init=False, default_factory=set)
    _handler_deferral_details: Dict[str, Dict[str, Any]] = field(
        init=False, default_factory=dict
    )
    _explicit_hydration_request: bool = field(init=False, default=False)
    _background_hydration_started: bool = field(init=False, default=False)
    _background_hydration_thread: threading.Thread | None = field(
        init=False, default=None
    )
    _background_handler_timeouts: Mapping[str, float] | float | None = field(
        init=False, default=None
    )
    _embedder_future: Future | None = field(init=False, default=None)
    _embedder_future_lock: threading.RLock | None = field(init=False, default=None)
    _embedder_deferred: bool = field(init=False, default=False)


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
        requested_hydration = self.hydrate_handlers
        if self.hydrate_handlers is None or (resolved_fast or warmup_lite):
            self.hydrate_handlers = False
        explicit_hydration = requested_hydration is True
        self._explicit_hydration_request = explicit_hydration
        if self.lazy_vector_store is None:
            self.lazy_vector_store = bootstrap_context or resolved_fast
        if warmup_lite:
            self.lazy_vector_store = True
        self._handler_bootstrap_flag = resolved_fast
        self._handlers = {}
        self._handler_requires_store = {}
        self._handler_lock = threading.RLock()
        self._vector_store_lock = threading.RLock()
        self._embedder_future_lock = threading.RLock()
        self._known_kinds = set()
        eager_hydration = bool(self.hydrate_handlers) and not (
            (warmup_lite or resolved_fast) and not explicit_hydration
        )
        if bootstrap_context and resolved_fast:
            logger.info(
                "shared_vector_service.bootstrap_fast.active",
                extra={
                    "bootstrap_fast_defaulted": defaulted_fast,
                    "bootstrap_context": True,
                },
            )
        self._known_kinds = set(_VECTOR_REGISTRY.keys())

        def _coerce_timeout(value: object) -> float | None:
            try:
                return float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return None

        inline_cap = _HEAVY_STAGE_CEILING
        handler_budget = None
        if isinstance(self.handler_timeouts, Mapping):
            handler_budget = _coerce_timeout(self.handler_timeouts.get("budget"))
        elif self.handler_timeouts is not None:
            handler_budget = _coerce_timeout(self.handler_timeouts)
        if handler_budget is not None:
            inline_cap = handler_budget if inline_cap is None else min(inline_cap, handler_budget)
            self._background_handler_timeouts = {"budget": handler_budget}
        elif inline_cap is not None:
            self._background_handler_timeouts = {"budget": inline_cap}
        else:
            self._background_handler_timeouts = self.handler_timeouts

        estimated_inline_cost = len(self._known_kinds) * _HANDLER_VECTOR_MIN_BUDGET
        inline_gate_exceeded = inline_cap is not None and estimated_inline_cost > inline_cap
        if inline_gate_exceeded:
            eager_hydration = False

        if eager_hydration:
            self.hydrate_all_handlers(
                stop_event=self.stop_event,
                handler_timeouts=self.handler_timeouts,
            )
        else:
            deferral_reason = (
                "inline-budget"
                if inline_gate_exceeded
                else (
                    "warmup-lite"
                    if warmup_lite
                    else "bootstrap-fast" if resolved_fast else "deferred-init"
                )
            )
            self._mark_handler_deferral(
                deferral_reason,
                schedule_background=True,
                timeout_hint=inline_cap,
                estimated_cost=estimated_inline_cost if inline_gate_exceeded else None,
            )
            logger.info(
                "shared_vector_service.handlers.deferred",
                extra=_timestamp_payload(
                    handler_start,
                    handler_count=len(self._known_kinds),
                    bootstrap_fast_active=resolved_fast,
                    bootstrap_context=bootstrap_context,
                    warmup_lite=warmup_lite,
                    inline_budget_cap=inline_cap,
                    estimated_inline_cost=estimated_inline_cost,
                    inline_gate_exceeded=inline_gate_exceeded,
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

    @property
    def handler_deferrals(self) -> Dict[str, Dict[str, Any]]:
        return {
            kind: dict(details) for kind, details in self._handler_deferral_details.items()
        }

    @property
    def handlers_ready(self) -> bool:
        return self._handlers_ready_flag or bool(self._handlers)

    def schedule_background_hydration(self) -> None:
        """Expose a best-effort background hydration trigger."""

        self._schedule_background_hydration()

    def kick_off_background_hydration(self) -> None:
        """Start handler hydration once the service is considered ready."""

        if self._handler_bootstrap_flag:
            self._handler_bootstrap_flag = False
        if self.warmup_lite:
            self.warmup_lite = False
        self._schedule_background_hydration()

    def hydrate_all_handlers(
        self,
        *,
        stop_event: threading.Event | None = None,
        timeout: float | None = None,
        budget_check: Callable[[threading.Event | None], None] | None = None,
        handler_timeouts: Mapping[str, float] | float | None = None,
    ) -> None:
        """Instantiate and cache all registered handlers."""

        handler_start = time.perf_counter()
        requested_timeouts = (
            handler_timeouts if handler_timeouts is not None else self.handler_timeouts
        )

        def _coerce_timeout(value: object) -> float | None:
            try:
                return float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return None

        effective_timeouts: Mapping[str, float] | float | None = requested_timeouts
        shared_budget = timeout
        if isinstance(requested_timeouts, Mapping):
            provided_budget = _coerce_timeout(requested_timeouts.get("budget"))
            if provided_budget is not None:
                shared_budget = (
                    min(shared_budget, provided_budget)
                    if shared_budget is not None
                    else provided_budget
                )
            elif shared_budget is not None:
                merged = dict(requested_timeouts)
                merged["budget"] = shared_budget
                effective_timeouts = merged
        elif requested_timeouts is None:
            if shared_budget is not None:
                effective_timeouts = {"budget": shared_budget}
        else:
            numeric_timeout = _coerce_timeout(requested_timeouts)
            if shared_budget is not None and numeric_timeout is not None:
                effective_timeouts = min(shared_budget, numeric_timeout)
            else:
                effective_timeouts = numeric_timeout if numeric_timeout is not None else None

        self._background_handler_timeouts = effective_timeouts
        if self.warmup_lite or self._handler_bootstrap_flag:
            reason = "warmup-lite" if self.warmup_lite else "bootstrap-fast"
            self._mark_handler_deferral(reason, schedule_background=True)
            logger.info(
                "shared_vector_service.handlers.deferred",
                extra={
                    "reason": reason,
                    "bootstrap_fast_active": self._handler_bootstrap_flag,
                    "explicit_hydration_request": self._explicit_hydration_request,
                },
            )
            return
        handler_list_timeout = timeout
        remaining_budget: float | None = None
        if handler_list_timeout is not None:
            remaining_budget = handler_list_timeout
        handlers: HandlerLoadResult = HandlerLoadResult()
        timed_out = False
        try:
            self._check_budget_deadline(
                "hydrate-handlers", handler_start, timeout, stop_event, budget_check
            )
            handlers = load_handlers(
                bootstrap_fast=self._handler_bootstrap_flag,
                warmup_lite=bool(self.warmup_lite),
                handler_timeouts=effective_timeouts,
                stop_event=stop_event or self.stop_event,
                budget_check=budget_check or self.budget_check,
            )
        except TimeoutError:
            timed_out = True

        deferral_statuses = getattr(handlers, "deferral_statuses", {})
        deferral_budgets = getattr(handlers, "deferral_budgets", {})
        processed: set[str] = set()
        handler_items = list(handlers.items())
        for index, (kind, handler) in enumerate(handler_items):
            self._handlers[kind] = handler
            processed.add(kind)
            self._handler_requires_store[kind] = self._handler_requires_store.get(
                kind, self._handler_requires_store_flag(handler)
            )
            try:
                self._check_budget_deadline(
                    "hydrate-handlers",
                    handler_start,
                    timeout,
                    stop_event,
                    budget_check,
                )
            except TimeoutError:
                timed_out = True
                remaining_budget = (
                    None
                    if timeout is None
                    else max(0.0, timeout - (time.perf_counter() - handler_start))
                )
                remaining_handlers = handler_items[index + 1 :]
                for pending_kind, _ in remaining_handlers:
                    deferral_statuses[pending_kind] = deferral_statuses.get(
                        pending_kind, "timeout"
                    )
                    if remaining_budget is not None:
                        deferral_budgets.setdefault(pending_kind, remaining_budget)
                break

        pending_known = [
            kind
            for kind in self._known_kinds
            if kind not in self._handlers and kind not in deferral_statuses
        ]
        if timed_out:
            for kind in pending_known:
                deferral_statuses[kind] = "timeout"
                if remaining_budget is not None:
                    deferral_budgets.setdefault(kind, remaining_budget)
            if deferral_statuses and not handlers:
                logger.info(
                    "shared_vector_service.handlers.deferred",
                    extra={"reason": "timeout"},
                )

        self._handler_deferral_details = {
            kind: {
                "reason": reason,
                "remaining_budget": deferral_budgets.get(kind),
            }
            for kind, reason in deferral_statuses.items()
        }
        if deferral_statuses:
            self._handler_hydration_deferred = True
            self._handler_deferral_reasons.update(deferral_statuses.values())
            self._schedule_background_hydration()
        else:
            self._clear_handler_deferral()
        if self._handlers or deferral_statuses:
            self._handlers_ready_flag = True
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
        if force and self.lazy_vector_store and self._should_skip_vector_store():
            logger.info(
                "shared_vector_service.vector_store.lazy_deferred",
                extra={
                    "bootstrap_fast": bool(self.bootstrap_fast),
                    "warmup_lite": bool(self.warmup_lite),
                },
            )
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
        self,
        reason: str,
        *,
        schedule_background: bool = False,
        timeout_hint: float | None = None,
        estimated_cost: float | None = None,
    ) -> None:
        self._handler_hydration_deferred = True
        self._handlers_ready_flag = True
        self._handler_deferral_reasons.add(reason)
        if _is_warmup_thread():
            meta: Dict[str, Any] = {"reason": reason}
            if timeout_hint is not None:
                meta["background_timeout"] = timeout_hint
            if estimated_cost is not None:
                meta["estimated_cost"] = estimated_cost
            _update_warmup_stage_cache(
                "handlers",
                reason,
                logger,
                meta={**meta, "background_state": _BACKGROUND_QUEUE_FLAG},
                emit_metric=False,
            )
        if schedule_background:
            self._schedule_background_hydration()

    def _clear_handler_deferral(self) -> None:
        self._handler_hydration_deferred = False
        self._handler_deferral_reasons.clear()
        self._handler_deferral_details.clear()
        self._handlers_ready_flag = bool(self._handlers)

    def _schedule_background_hydration(self) -> None:
        if self._background_hydration_started:
            if (
                self._background_hydration_thread is not None
                and not self._background_hydration_thread.is_alive()
            ):
                self._background_hydration_started = False
            else:
                return
        if self.stop_event is not None and self.stop_event.is_set():
            return
        self._background_hydration_started = True

        def _runner() -> None:
            try:
                self.hydrate_all_handlers(
                    stop_event=self.stop_event,
                    handler_timeouts=self._background_handler_timeouts,
                    budget_check=self.budget_check,
                )
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

    def _probe_embedder_state(self) -> tuple[bool, bool]:
        """Return embedder availability and placeholder state without loading it."""

        embedder_available = self.text_embedder is not None
        placeholder_present = False
        future = self._embedder_future
        if future is not None and not future.cancelled() and future.done():
            try:
                embedder_available = embedder_available or future.result() is not None
            except Exception:
                pass
        try:
            embedder_available = embedder_available or bool(
                getattr(governed_embeddings, "_EMBEDDER", None)
            )
            placeholder_present = bool(
                getattr(governed_embeddings, "_EMBEDDER_BOOTSTRAP_PLACEHOLDER", None)
            )
        except Exception:
            pass
        if not embedder_available and SentenceTransformer is not None:
            embedder_available = True
        return embedder_available, placeholder_present

    def probe_text_embedder(self) -> tuple[bool, bool]:
        """Expose embedder readiness without triggering model downloads."""

        return self._probe_embedder_state()

    def _record_embedder_metric(self, status: str) -> None:
        try:
            VECTOR_EMBEDDER_RESOLVE_TOTAL.labels(status).inc()
        except Exception:  # pragma: no cover - metrics best effort
            logger.debug("failed emitting embedder metric", exc_info=True)

    def _trigger_background_embedder_download(self) -> None:
        try:
            ensure_embedding_model_future(
                logger=logger,
                warmup=True,
                warmup_lite=False,
                warmup_heavy=True,
                download_timeout=_REMOTE_TIMEOUT,
            )
        except Exception:  # pragma: no cover - background best effort
            logger.debug("failed to trigger background embedder download", exc_info=True)

    def _build_deferred_placeholder(self) -> SentenceTransformer | None:
        builder = getattr(governed_embeddings, "_build_stub_embedder", None)
        if not callable(builder):
            return None
        placeholder = builder()
        setattr(placeholder, "_placeholder_reason", "deferred-ceiling")
        return placeholder

    def _get_embedder_future(self, *, force: bool, bootstrap_mode: bool) -> Future:
        lock = self._embedder_future_lock
        if lock is None:
            lock = threading.RLock()
            self._embedder_future_lock = lock
        with lock:
            future = self._embedder_future
            if future is not None and not future.cancelled():
                if future.done():
                    try:
                        result = future.result()
                    except Exception:
                        if not force:
                            return future
                        future = None
                    else:
                        if result is not None:
                            if self.text_embedder is None:
                                self.text_embedder = result
                            return future
                        if not force:
                            return future
                        future = None
                else:
                    return future

            future = Future()

            def _resolve() -> None:
                if not future.set_running_or_notify_cancel():
                    return
                try:
                    embedder = get_embedder(
                        timeout=0.0 if (bootstrap_mode and not force) else None,
                        bootstrap_timeout=0.0 if (bootstrap_mode and not force) else None,
                        bootstrap_mode=bootstrap_mode,
                        stop_event=self.stop_event,
                    )
                except Exception as exc:  # pragma: no cover - defensive logging
                    future.set_exception(exc)
                else:
                    future.set_result(embedder)

            thread = threading.Thread(
                target=_resolve,
                name="shared-vector-service-embedder",
                daemon=True,
            )
            thread.start()
            self._embedder_future = future
            return future

    def _wait_for_embedder_future(
        self,
        future: Future,
        *,
        timeout: float | None,
        stop_event: threading.Event | None,
    ) -> SentenceTransformer | None:
        deadline = None if timeout is None else time.perf_counter() + timeout
        while True:
            if stop_event is not None and stop_event.is_set():
                if not future.done():
                    future.cancel()
                raise TimeoutError("shared vector service embedder cancelled")
            try:
                wait_timeout = None
                if deadline is not None:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        raise TimeoutError("shared vector service embedder timed out")
                    wait_timeout = max(min(remaining, 0.1), 0.0)
                return future.result(timeout=wait_timeout)
            except FutureTimeout:
                continue

    def _ensure_text_embedder(self, *, force: bool = False) -> SentenceTransformer | None:
        """Initialise ``self.text_embedder`` if possible, respecting warmup guards."""

        if self.text_embedder is not None:
            return self.text_embedder
        if SentenceTransformer is None:
            return None

        self._embedder_deferred = False
        bootstrap_mode = bool(self.bootstrap_fast or self.warmup_lite)
        future = self._get_embedder_future(force=force, bootstrap_mode=bootstrap_mode)
        wait_timeout = 0.0 if (bootstrap_mode and not force) else None
        ceiling = None if bootstrap_mode else _EMBEDDER_WAIT_CEILING
        if wait_timeout is None and ceiling is not None:
            wait_timeout = ceiling
        stop_event = self.stop_event
        _trace(
            "shared_vector_service.embedder.resolve.start",
            bootstrap_mode=bootstrap_mode,
            ceiling=wait_timeout,
        )
        try:
            embedder = self._wait_for_embedder_future(
                future, timeout=wait_timeout, stop_event=stop_event
            )
        except TimeoutError:
            self._embedder_deferred = True
            placeholder = self._build_deferred_placeholder()
            self._record_embedder_metric("deferred")
            self._trigger_background_embedder_download()
            _trace(
                "shared_vector_service.embedder.resolve.deferred",
                ceiling=wait_timeout,
                placeholder=bool(placeholder),
            )
            return placeholder
        except Exception as exc:  # pragma: no cover - defensive logging
            if getattr(exc, "_deferred_status", None) == _BACKGROUND_QUEUE_FLAG:
                self._embedder_deferred = True
                placeholder = self._build_deferred_placeholder()
                self._record_embedder_metric("deferred")
                _trace(
                    "shared_vector_service.embedder.resolve.deferred",
                    ceiling=wait_timeout,
                    placeholder=bool(placeholder),
                )
                return placeholder
            _trace(
                "shared_vector_service.embedder.resolve.failed",
                error=str(exc),
            )
            self._record_embedder_metric("failed")
            return None
        if embedder is None:
            _trace("shared_vector_service.embedder.resolve.failed")
            self._record_embedder_metric("failed")
            return None
        self.text_embedder = embedder
        self._record_embedder_metric("success")
        _trace("shared_vector_service.embedder.resolve.success")
        return embedder

    def _encode_text(
        self,
        text: str,
        *,
        kind: str | None = None,
        original_text: str | None = None,
    ) -> List[float]:
        raw_text = text if original_text is None else original_text
        normalized = _normalize_redacted_text(text)
        if not normalized:
            short_hash = hashlib.sha256(
                str(raw_text).encode("utf-8", "ignore")
            ).hexdigest()[:8]
            logger.warning(
                "empty input after redaction; returning zero vector",
                extra={"kind": kind, "text_hash": short_hash},
            )
            embedder = self.text_embedder
            if embedder is None and SentenceTransformer is not None:
                embedder = self._ensure_text_embedder(force=False)
            dimension = 0
            if embedder is not None:
                dimension = getattr(
                    embedder, "get_sentence_embedding_dimension", lambda: 0
                )()
            if dimension <= 0:
                return []
            return [0.0] * int(dimension)
        text = normalized
        embedder = self._ensure_text_embedder(force=True)
        embedder_available = embedder is not None
        placeholder_reason = getattr(embedder, "_placeholder_reason", None)
        if not embedder_available and self._embedder_deferred:
            _trace(
                "shared_vector_service.encode_text.embedder.deferred",
                ceiling=_EMBEDDER_WAIT_CEILING,
            )
            raise RuntimeError("text embedder initialisation deferred")

        if embedder_available or SentenceTransformer is not None:
            _trace(
                "shared_vector_service.encode_text.embedder",
                source="governed",
                embedder_available=embedder_available,
                placeholder_reason=placeholder_reason,
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
        global _REMOTE_DISABLED, _REMOTE_BOOT_SKIP_LOGGED
        if _REMOTE_DISABLED or _REMOTE_ENDPOINT is None:
            return None
        skip_secs = _remote_boot_skip_secs()
        if skip_secs:
            elapsed = time.monotonic() - _VECTOR_SERVICE_BOOT_TS
            if elapsed < skip_secs:
                if not _REMOTE_BOOT_SKIP_LOGGED:
                    logger.info(
                        "remote vector service skipped during boot grace period",
                        extra={
                            "elapsed_secs": round(elapsed, 3),
                            "skip_secs": skip_secs,
                        },
                    )
                    _REMOTE_BOOT_SKIP_LOGGED = True
                return None
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{_REMOTE_ENDPOINT}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        attempts = _REMOTE_ATTEMPTS
        delay = _REMOTE_RETRY_DELAY
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                with urllib.request.urlopen(
                    req, timeout=_REMOTE_TIMEOUT or None
                ) as resp:  # pragma: no cover - network
                    return json.loads(resp.read().decode("utf-8"))
            except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
                last_exc = exc
                logger.debug(
                    "remote vector service attempt %s/%s failed: %s",
                    attempt,
                    attempts,
                    exc,
                )
                if attempt < attempts:
                    time.sleep(delay)
                    continue
            except json.JSONDecodeError as exc:
                logger.warning(
                    "remote vector service returned invalid JSON; falling back to local handling",  # pragma: no cover - diagnostics
                    extra={"endpoint": _REMOTE_ENDPOINT, "path": path, "error": str(exc)},
                )
                _REMOTE_DISABLED = True
                return None

        exc = last_exc or RuntimeError("remote vector service unavailable")
        if _wait_for_remote_ready_short():
            try:
                with urllib.request.urlopen(
                    req, timeout=_REMOTE_TIMEOUT or None
                ) as resp:  # pragma: no cover - network
                    return json.loads(resp.read().decode("utf-8"))
            except (urllib.error.URLError, TimeoutError, socket.timeout) as retry_exc:
                exc = retry_exc
            except json.JSONDecodeError as retry_exc:
                logger.warning(
                    "remote vector service returned invalid JSON; falling back to local handling",  # pragma: no cover - diagnostics
                    extra={
                        "endpoint": _REMOTE_ENDPOINT,
                        "path": path,
                        "error": str(retry_exc),
                    },
                )
                _REMOTE_DISABLED = True
                return None
        if _wait_for_remote_ready():
            try:
                with urllib.request.urlopen(
                    req, timeout=_REMOTE_TIMEOUT or None
                ) as resp:  # pragma: no cover - network
                    return json.loads(resp.read().decode("utf-8"))
            except (urllib.error.URLError, TimeoutError, socket.timeout) as retry_exc:
                exc = retry_exc
            except json.JSONDecodeError as retry_exc:
                logger.warning(
                    "remote vector service returned invalid JSON; falling back to local handling",  # pragma: no cover - diagnostics
                    extra={
                        "endpoint": _REMOTE_ENDPOINT,
                        "path": path,
                        "error": str(retry_exc),
                    },
                )
                _REMOTE_DISABLED = True
                return None
        if _wait_for_remote_ready_cold_start():
            try:
                with urllib.request.urlopen(
                    req, timeout=_REMOTE_TIMEOUT or None
                ) as resp:  # pragma: no cover - network
                    return json.loads(resp.read().decode("utf-8"))
            except (urllib.error.URLError, TimeoutError, socket.timeout) as retry_exc:
                exc = retry_exc
            except json.JSONDecodeError as retry_exc:
                logger.warning(
                    "remote vector service returned invalid JSON; falling back to local handling",  # pragma: no cover - diagnostics
                    extra={
                        "endpoint": _REMOTE_ENDPOINT,
                        "path": path,
                        "error": str(retry_exc),
                    },
                )
                _REMOTE_DISABLED = True
                return None
        logger.warning(
            "remote vector service unavailable; falling back to local handling",  # pragma: no cover - diagnostics
            extra={"endpoint": _REMOTE_ENDPOINT, "path": path, "error": str(exc)},
        )
        _REMOTE_DISABLED = True
        return None

    def ready(self) -> bool:
        """Return ``True`` when the service can respond to a request."""

        if _REMOTE_ENDPOINT is not None and not _REMOTE_DISABLED:
            return True
        if self.handlers_ready:
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
            raw_text = str(record.get("text", ""))
            return self._encode_text(raw_text, kind=kind, original_text=raw_text)
        if kind in {"stack"}:
            _trace("shared_vector_service.vectorise.stack", kind=kind)
            raw_text = str(record.get("text", ""))
            return self._encode_text(raw_text, kind=kind, original_text=raw_text)
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
