from __future__ import annotations

"""Lightweight SQLite store for vector operation metrics.

Warmup and bootstrap flows return an in-memory stub by default so callers do
not touch SQLite or the dynamic path router until after readiness is signaled
or the first real metric is written.  A background readiness hook can promote
the stub once bootstrap completes, while the stub also arms a first-write
activation path so normal callers eventually create the real database even if
no readiness signal is emitted.  Pending weight seeds are replayed when the
stub is promoted so bootstrap-time configuration is preserved.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple, Mapping, Sequence, TYPE_CHECKING
import threading
import contextlib
import json
import logging
import os
import time
import sqlite3

if TYPE_CHECKING:  # pragma: no cover - import hints only
    import db_router as _db_router_module
    import dynamic_path_router as _dynamic_path_router

try:  # pragma: no cover - optional dependency
    from . import metrics_exporter as _me
except Exception:  # pragma: no cover - fallback when running directly
    import metrics_exporter as _me  # type: ignore

# Prometheus gauges/counters (instantiated lazily)
_EMBEDDING_TOKENS_TOTAL = None
_RETRIEVAL_HIT_RATE = None
_RETRIEVER_WIN_RATE = None
_RETRIEVER_REGRET_RATE = None


logger = logging.getLogger(__name__)

_DB_ROUTER_MODULE: "_db_router_module | None" = None
_DYNAMIC_PATH_ROUTER: "_dynamic_path_router | None" = None

_VECTOR_DB_INSTANCE: "VectorMetricsDB | _BootstrapVectorMetricsStub | None" = None
_VECTOR_DB_LOCK = threading.Lock()
_PENDING_WEIGHTS: dict[str, float] = {}
_READINESS_HOOK_ARMED = False

_BOOTSTRAP_TIMER_ENVS = (
    "MENACE_BOOTSTRAP_WAIT_SECS",
    "MENACE_BOOTSTRAP_VECTOR_WAIT_SECS",
    "BOOTSTRAP_STEP_TIMEOUT",
    "BOOTSTRAP_VECTOR_STEP_TIMEOUT",
    "PREPARE_PIPELINE_VECTORIZER_BUDGET_SECS",
    "PREPARE_PIPELINE_RETRIEVER_BUDGET_SECS",
    "PREPARE_PIPELINE_DB_WARMUP_BUDGET_SECS",
    "PREPARE_PIPELINE_ORCHESTRATOR_BUDGET_SECS",
    "PREPARE_PIPELINE_CONFIG_BUDGET_SECS",
)


class _BootstrapVectorMetricsStub:
    """Lightweight placeholder that defers VectorMetricsDB creation."""

    def __init__(
        self,
        *,
        path: str | Path,
        bootstrap_fast: bool | None,
        warmup: bool | None,
        ensure_exists: bool | None,
        read_only: bool | None,
    ) -> None:
        self._activation_kwargs = {
            "path": path,
            "bootstrap_fast": False,
            "warmup": False,
            "ensure_exists": ensure_exists,
            "read_only": read_only,
        }
        if bootstrap_fast is not None:
            self._activation_kwargs["bootstrap_fast"] = bool(bootstrap_fast)
        if warmup is not None:
            self._activation_kwargs["warmup"] = bool(warmup)
        self._delegate: "VectorMetricsDB | None" = None
        self._noop_calls: set[str] = set()
        self._readiness_hook_registered = False
        self._activate_on_first_write = False
        self._deferred_summary_emitted = False
        self._pending_weights: dict[str, float] = {}
        self._queued_first_write = False
        self._queued_activation_kwargs: dict[str, Any] = {}
        self._activation_blocked = bool(
            self._activation_kwargs.get("bootstrap_fast")
            or self._activation_kwargs.get("warmup")
        )
        logger.info(
            "vector_metrics_db.bootstrap.stub_created",
            extra={
                "bootstrap_fast": bool(self._activation_kwargs.get("bootstrap_fast")),
                "warmup": bool(self._activation_kwargs.get("warmup")),
                "read_only": bool(self._activation_kwargs.get("read_only")),
                "ensure_exists": self._activation_kwargs.get("ensure_exists"),
            },
        )

    @property
    def _boot_stub_active(self) -> bool:  # pragma: no cover - passthrough flag
        return self._delegate is None

    def activate_on_first_write(self) -> None:
        if self._delegate is not None:
            self._delegate.activate_on_first_write()
            return
        if self._activation_blocked:
            self._queued_first_write = True
            self._log_deferred_activation(reason="activate_on_first_write_blocked")
            return
        self._activate_on_first_write = True
        logger.info(
            "vector_metrics_db.bootstrap.activation_deferred",
            extra={
                "bootstrap_fast": bool(self._activation_kwargs.get("bootstrap_fast")),
                "warmup": bool(self._activation_kwargs.get("warmup")),
                "read_only": bool(self._activation_kwargs.get("read_only")),
                "ensure_exists": self._activation_kwargs.get("ensure_exists"),
            },
        )

    def configure_activation(
        self,
        *,
        bootstrap_fast: bool | None = None,
        warmup: bool | None = None,
        ensure_exists: bool | None = None,
        read_only: bool | None = None,
    ) -> None:
        if self._delegate is not None:
            return
        warmup_context = bool(
            self._activation_kwargs.get("bootstrap_fast")
            or self._activation_kwargs.get("warmup")
        )
        updates: dict[str, Any] = {}
        if bootstrap_fast is not None:
            updates["bootstrap_fast"] = bool(bootstrap_fast)
        if warmup is not None:
            if warmup_context and warmup:
                self._log_deferred_activation(reason="warmup_request")
            else:
                updates["warmup"] = bool(warmup)
        if ensure_exists is not None:
            if warmup_context and ensure_exists:
                self._log_deferred_activation(reason="ensure_exists_request")
            else:
                updates["ensure_exists"] = ensure_exists
        if read_only is not None:
            updates["read_only"] = read_only
        if updates:
            self._activation_kwargs.update(updates)
        if self._activation_blocked:
            self._queued_activation_kwargs.update(
                {k: v for k, v in self._activation_kwargs.items() if k in updates}
            )
        else:
            self._activation_blocked = bool(
                self._activation_kwargs.get("bootstrap_fast")
                or self._activation_kwargs.get("warmup")
            )

    def _log_deferred_activation(self, *, reason: str) -> None:
        if self._delegate is not None:
            return
        if self._deferred_summary_emitted:
            return
        self._deferred_summary_emitted = True
        logger.info(
            "vector_metrics_db.bootstrap.activation_deferred",
            extra={
                "reason": reason,
                "bootstrap_fast": bool(self._activation_kwargs.get("bootstrap_fast")),
                "warmup": bool(self._activation_kwargs.get("warmup")),
                "read_only": bool(self._activation_kwargs.get("read_only")),
                "ensure_exists": self._activation_kwargs.get("ensure_exists"),
            },
        )

    def _release_activation_block(self, *, reason: str, configure_ready: bool) -> None:
        if not self._activation_blocked:
            return
        if configure_ready:
            self._activation_kwargs["warmup"] = False
            self._activation_kwargs.setdefault("ensure_exists", True)
            self._activation_kwargs.setdefault("read_only", False)
        if self._queued_activation_kwargs:
            self._activation_kwargs.update(self._queued_activation_kwargs)
            self._queued_activation_kwargs.clear()
        if self._queued_first_write:
            self._activate_on_first_write = True
        self._activation_blocked = False
        logger.info(
            "vector_metrics_db.bootstrap.activation_unblocked",
            extra={
                "reason": reason,
                "bootstrap_fast": bool(self._activation_kwargs.get("bootstrap_fast")),
                "warmup": bool(self._activation_kwargs.get("warmup")),
                "read_only": bool(self._activation_kwargs.get("read_only")),
            },
        )

    def activate(self, *, reason: str = "explicit") -> "VectorMetricsDB":
        return self._activate(reason=reason)

    def register_readiness_hook(self) -> None:
        if self._delegate is not None:
            return
        if self._readiness_hook_registered:
            return
        try:  # pragma: no cover - optional dependency
            from bootstrap_readiness import readiness_signal
        except Exception:
            logger.debug("vector metrics stub readiness hook unavailable", exc_info=True)
            return

        self._readiness_hook_registered = True

        def _await_ready() -> None:
            try:
                readiness_signal().await_ready(timeout=None)
            except Exception:  # pragma: no cover - best effort logging
                logger.debug(
                    "vector metrics stub readiness gate failed", exc_info=True
                )
                return
            try:
                activate_shared_vector_metrics_db(
                    reason="bootstrap_ready", post_warmup=True
                )
            except Exception:  # pragma: no cover - best effort logging
                logger.debug(
                    "vector metrics stub readiness activation failed", exc_info=True
                )

        logger.info(
            "vector_metrics_db.bootstrap.stub_readiness_registered",
            extra={
                "bootstrap_fast": bool(self._activation_kwargs.get("bootstrap_fast")),
                "warmup": bool(self._activation_kwargs.get("warmup")),
                "read_only": bool(self._activation_kwargs.get("read_only")),
                "ensure_exists": self._activation_kwargs.get("ensure_exists"),
            },
        )
        threading.Thread(
            target=_await_ready,
            name="vector-metrics-stub-readiness",
            daemon=True,
        ).start()

    def _promote_from_stub(self, *, reason: str) -> "VectorMetricsDB | None":
        delegate = self._delegate
        if delegate is not None:
            return delegate
        if self._activation_blocked:
            allow_unblock = reason in {"bootstrap_ready", "post_warmup", "warmup_complete"}
            if allow_unblock:
                self._release_activation_block(reason=reason, configure_ready=True)
            else:
                self._log_deferred_activation(reason=reason)
                return None
        delegate = self._activate(reason=reason)
        if isinstance(delegate, VectorMetricsDB):
            try:  # pragma: no cover - best effort promotion
                if reason == "bootstrap_ready":
                    delegate.end_warmup(reason=reason, activate=False)
                    delegate.activate_on_first_write()
                else:
                    delegate.end_warmup(reason=reason)
                    delegate.activate_persistence(reason=reason)
            except Exception:
                logger.debug(
                    "vector metrics stub promotion failed", exc_info=True
                )
            else:
                logger.info(
                    "vector_metrics_db.bootstrap.stub_promoted",
                    extra={
                        "reason": reason,
                        "bootstrap_fast": bool(
                            self._activation_kwargs.get("bootstrap_fast")
                        ),
                        "warmup": bool(self._activation_kwargs.get("warmup")),
                        "read_only": bool(self._activation_kwargs.get("read_only")),
                    },
                )
        return delegate

    def _activate(self, *, reason: str = "attribute_access") -> "VectorMetricsDB":
        if self._delegate is not None:
            return self._delegate
        if self._activation_blocked:
            self._log_deferred_activation(reason=reason)
            return self
        if self._pending_weights:
            _record_pending_weight_values(self._pending_weights)
        activation_kwargs = dict(self._activation_kwargs)
        activation_kwargs.setdefault("bootstrap_fast", False)
        activation_kwargs.setdefault("warmup", False)
        vdb = VectorMetricsDB(**activation_kwargs)
        if self._activate_on_first_write:
            vdb.activate_on_first_write()
        _apply_pending_weights(vdb)
        self._delegate = vdb
        global _VECTOR_DB_INSTANCE
        _VECTOR_DB_INSTANCE = vdb
        logger.info(
            "vector_metrics_db.bootstrap.stub_activated",
            extra={
                "reason": reason,
                "bootstrap_fast": bool(self._activation_kwargs.get("bootstrap_fast")),
                "warmup": bool(self._activation_kwargs.get("warmup")),
                "read_only": bool(self._activation_kwargs.get("read_only")),
            },
        )
        return vdb

    def _noop(self, *, method: str, default: Any = None):
        if method not in self._noop_calls:
            self._noop_calls.add(method)
            logger.info(
                "vector_metrics_db.bootstrap.stub_noop",
                extra={
                    "method": method,
                    "bootstrap_fast": bool(self._activation_kwargs.get("bootstrap_fast")),
                    "warmup": bool(self._activation_kwargs.get("warmup")),
                    "read_only": bool(self._activation_kwargs.get("read_only")),
                },
            )
        return default

    def _activate_for_write(self, *, method: str):
        delegate = self._delegate
        if delegate is not None:
            return delegate
        if self._activation_blocked:
            return None
        if not self._activate_on_first_write:
            return None
        delegate = self._activate(reason=f"write:{method}")
        if delegate is self:
            return None
        return delegate

    def planned_path(self) -> Path:
        return Path(self._activation_kwargs["path"]).expanduser()

    def ready_probe(self) -> str:
        return str(self.planned_path())

    def persistence_probe(self) -> bool:
        return False

    def set_db_weights(self, weights: Mapping[str, float]):
        _record_pending_weight_values(weights)
        self._pending_weights.update({str(k): float(v) for k, v in weights.items()})
        if self._delegate is None:
            return None
        return self._delegate.set_db_weights(weights)

    def get_db_weights(self, default: Mapping[str, float] | None = None):
        if self._delegate is None:
            pending = _pending_weight_mapping()
            if pending:
                return pending
            return default or {}
        return self._delegate.get_db_weights(default=default)

    def add(self, *_args, **_kwargs):
        delegate = self._activate_for_write(method="add")
        if delegate is None:
            return self._noop(method="add")
        return delegate.add(*_args, **_kwargs)

    def log_embedding(self, *_args, **_kwargs):
        delegate = self._activate_for_write(method="log_embedding")
        if delegate is None:
            return self._noop(method="log_embedding")
        return delegate.log_embedding(*_args, **_kwargs)

    def log_retrieval(self, *_args, **_kwargs):
        delegate = self._activate_for_write(method="log_retrieval")
        if delegate is None:
            return self._noop(method="log_retrieval")
        return delegate.log_retrieval(*_args, **_kwargs)

    def log_retrieval_feedback(self, *_args, **_kwargs):
        delegate = self._activate_for_write(method="log_retrieval_feedback")
        if delegate is None:
            return self._noop(method="log_retrieval_feedback")
        return delegate.log_retrieval_feedback(*_args, **_kwargs)

    def log_ranker_update(self, *_args, **_kwargs):
        delegate = self._activate_for_write(method="log_ranker_update")
        if delegate is None:
            return self._noop(method="log_ranker_update")
        return delegate.log_ranker_update(*_args, **_kwargs)

    def save_session(self, *_args, **_kwargs):
        delegate = self._activate_for_write(method="save_session")
        if delegate is None:
            return self._noop(method="save_session")
        return delegate.save_session(*_args, **_kwargs)

    def load_sessions(self, *_args, **_kwargs):
        delegate = self._activate_for_write(method="load_sessions")
        if delegate is None:
            return self._noop(method="load_sessions", default={})
        return delegate.load_sessions(*_args, **_kwargs)

    def delete_session(self, *_args, **_kwargs):
        delegate = self._activate_for_write(method="delete_session")
        if delegate is None:
            return self._noop(method="delete_session")
        return delegate.delete_session(*_args, **_kwargs)

    def embedding_tokens_total(self, *_args, **_kwargs):
        if self._delegate is None:
            return self._noop(method="embedding_tokens_total", default=0)
        return self._delegate.embedding_tokens_total(*_args, **_kwargs)

    def activate_persistence(self, *, reason: str = "stub_ready") -> "VectorMetricsDB | None":
        delegate = self._promote_from_stub(reason=reason)
        if delegate is None:
            return None
        return delegate

    def activate_router(self, *, reason: str = "stub_ready") -> "VectorMetricsDB | None":
        delegate = self._promote_from_stub(reason=reason)
        if isinstance(delegate, VectorMetricsDB):
            return delegate.activate_router(reason=reason)
        return delegate

    def __getattr__(self, name: str):
        if self._delegate is not None:
            return getattr(self._delegate, name)
        noop_prefixes = (
            "log_",
            "add",
            "save_",
            "load_",
            "delete_",
            "embedding_",
        )
        if name.startswith(noop_prefixes):
            return lambda *args, **kwargs: self._noop(method=name)
        if self._activation_blocked:
            return lambda *args, **kwargs: self._noop(method=name)
        delegate = self._activate(reason=f"attr:{name}")
        return getattr(delegate, name)


def _db_router():
    global _DB_ROUTER_MODULE
    if _DB_ROUTER_MODULE is None:  # pragma: no cover - lazy import
        import db_router as db_router_module

        _DB_ROUTER_MODULE = db_router_module
    return _DB_ROUTER_MODULE


def _dynamic_path_router():
    global _DYNAMIC_PATH_ROUTER
    if _DYNAMIC_PATH_ROUTER is None:  # pragma: no cover - lazy import
        import dynamic_path_router as dynamic_path_router_module

        _DYNAMIC_PATH_ROUTER = dynamic_path_router_module
    return _DYNAMIC_PATH_ROUTER


def _record_pending_weights(names: Sequence[str], *, value: float = 1.0) -> None:
    clean_names = {str(n): float(value) for n in names if n}
    if not clean_names:
        return
    with _VECTOR_DB_LOCK:
        _PENDING_WEIGHTS.update(clean_names)


def _record_pending_weight_values(weights: Mapping[str, float]) -> None:
    if not weights:
        return
    with _VECTOR_DB_LOCK:
        for name, val in weights.items():
            if name:
                _PENDING_WEIGHTS[str(name)] = float(val)


def _pending_weight_mapping() -> dict[str, float]:
    with _VECTOR_DB_LOCK:
        return dict(_PENDING_WEIGHTS)


def _consume_pending_weights() -> dict[str, float]:
    with _VECTOR_DB_LOCK:
        pending = dict(_PENDING_WEIGHTS)
        return pending


def _clear_pending_weights(names: Sequence[str]) -> None:
    if not names:
        return
    with _VECTOR_DB_LOCK:
        for name in names:
            _PENDING_WEIGHTS.pop(str(name), None)


def _apply_pending_weights(vdb: "VectorMetricsDB") -> None:
    pending = _consume_pending_weights()
    if not pending:
        return
    if getattr(vdb, "_boot_stub_active", False):
        return

    try:
        existing = vdb.get_db_weights(default=_pending_weight_mapping())
    except Exception as exc:  # pragma: no cover - best effort
        logger.info(
            "vector_metrics_db.bootstrap.pending_weights_cached",
            extra={"count": len(pending), "reason": str(exc)},
        )
        return

    if not existing:
        try:
            vdb.set_db_weights(dict(pending))
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning(
                "vector_metrics_db.bootstrap.pending_weights_failed",
                exc_info=exc,
                extra={"count": len(pending)},
            )
            return
        _clear_pending_weights(pending.keys())
        return

    missing = {name: weight for name, weight in pending.items() if name not in existing}
    if not missing:
        _clear_pending_weights(existing.keys())
        return

    try:
        merged = dict(existing)
        merged.update(missing)
        vdb.set_db_weights(merged)
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning(
            "vector_metrics_db.bootstrap.pending_weights_partial_failure",
            exc_info=exc,
            extra={"count": len(missing)},
        )
        return
    _clear_pending_weights(merged.keys())


def _arm_shared_readiness_hook() -> None:
    """Register a readiness callback to promote the shared DB post-warmup."""

    global _READINESS_HOOK_ARMED
    if _READINESS_HOOK_ARMED:
        return
    try:  # pragma: no cover - optional dependency
        from bootstrap_readiness import readiness_signal
    except Exception:
        return

    _READINESS_HOOK_ARMED = True

    instance = _VECTOR_DB_INSTANCE
    if isinstance(instance, _BootstrapVectorMetricsStub):
        instance.register_readiness_hook()
        return

    def _await_ready() -> None:
        try:
            readiness_signal().await_ready(timeout=None)
        except Exception:  # pragma: no cover - best effort
            logger.debug("vector_metrics_db.bootstrap.readiness_gate_failed", exc_info=True)
            return
        try:
            activate_shared_vector_metrics_db(reason="bootstrap_ready")
        except Exception:  # pragma: no cover - best effort
            logger.debug(
                "vector_metrics_db.bootstrap.readiness_activation_failed", exc_info=True
            )

    threading.Thread(
        target=_await_ready,
        name="vector-metrics-shared-readiness",
        daemon=True,
    ).start()


def _ensure_prometheus_objects() -> tuple:
    """Instantiate Prometheus metrics on-demand.

    Gauge registration during process bootstrap can block when the collector
    registry is initialising, so we defer creation until the first actual
    metric write.  The gauges are cached globally once constructed.
    """

    global _EMBEDDING_TOKENS_TOTAL, _RETRIEVAL_HIT_RATE
    global _RETRIEVER_WIN_RATE, _RETRIEVER_REGRET_RATE

    if _EMBEDDING_TOKENS_TOTAL is None:
        _EMBEDDING_TOKENS_TOTAL = _me.Gauge(
            "embedding_tokens_total",
            "Total tokens processed for embeddings",
        )
    if _RETRIEVAL_HIT_RATE is None:
        _RETRIEVAL_HIT_RATE = _me.Gauge(
            "retrieval_hit_rate",
            "Fraction of retrieval results included in final prompt",
        )
    if _RETRIEVER_WIN_RATE is None:
        _RETRIEVER_WIN_RATE = getattr(
            _me,
            "retriever_win_rate",
            _me.Gauge(
                "retriever_win_rate",
                "Win rate of retrieval operations by database",
                ["db"],
            ),
        )
    if _RETRIEVER_REGRET_RATE is None:
        _RETRIEVER_REGRET_RATE = getattr(
            _me,
            "retriever_regret_rate",
            _me.Gauge(
                "retriever_regret_rate",
                "Regret rate of retrieval operations by database",
                ["db"],
            ),
        )
    return (
        _EMBEDDING_TOKENS_TOTAL,
        _RETRIEVAL_HIT_RATE,
        _RETRIEVER_WIN_RATE,
        _RETRIEVER_REGRET_RATE,
    )


def _timestamp_payload(start: float | None = None, **extra: Any) -> Dict[str, Any]:
    payload = {"ts": datetime.utcnow().isoformat(), **extra}
    if start is not None:
        payload["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 3)
    return payload


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _noop_logging(bootstrap_fast: bool, warmup_mode: bool) -> bool:
    return bool(bootstrap_fast or warmup_mode)


def _bootstrap_timers_active() -> bool:
    if _env_flag("VECTOR_METRICS_BOOTSTRAP_WARMUP", False):
        return True
    return any(os.getenv(name) for name in _BOOTSTRAP_TIMER_ENVS)


def _detect_bootstrap_environment() -> dict[str, bool]:
    vector_bootstrap_env = _env_flag("VECTOR_METRICS_BOOTSTRAP_FAST", False)
    patch_bootstrap_env = _env_flag("PATCH_HISTORY_BOOTSTRAP", False)
    vector_service_warmup = _env_flag("VECTOR_SERVICE_WARMUP", False)
    vector_warmup_env = _env_flag("VECTOR_WARMUP", False) or vector_service_warmup
    bootstrap_env = any(
        _env_flag(flag, False)
        for flag in (
            "MENACE_BOOTSTRAP",
            "MENACE_BOOTSTRAP_MODE",
            "MENACE_BOOTSTRAP_FAST",
        )
    )

    return {
        "vector_bootstrap_env": vector_bootstrap_env,
        "patch_bootstrap_env": patch_bootstrap_env,
        "vector_service_warmup": vector_service_warmup,
        "vector_warmup_env": vector_warmup_env,
        "bootstrap_env": bootstrap_env,
        "bootstrap_timers": _bootstrap_timers_active(),
    }


def _bootstrap_state_flags() -> dict[str, bool]:
    """Return bootstrap hints exposed by the coding bot bootstrap state."""

    state = getattr(sys.modules.get("coding_bot_interface"), "_BOOTSTRAP_STATE", None)
    try:
        warmup_lite = bool(getattr(state, "warmup_lite", False))
    except Exception:  # pragma: no cover - defensive
        warmup_lite = False
    return {"bootstrap_state": bool(state), "warmup_lite": warmup_lite}


def resolve_vector_bootstrap_flags(
    *, bootstrap_fast: bool | None = None, warmup: bool | None = None
) -> tuple[bool, bool, bool, bool]:
    """Resolve bootstrap flags using explicit values and environment hints.

    Returns a tuple ``(bootstrap_fast, warmup, env_requested, bootstrap_env)``
    where ``env_requested`` is ``True`` when any bootstrap or warmup env vars
    are present and ``bootstrap_env`` reflects the generic MENACE bootstrap
    flags.  Callers can use these values to avoid touching the filesystem
    during bootstrap while still allowing explicit overrides via arguments.
    """

    env = _detect_bootstrap_environment()

    env_requested = bool(
        env["vector_bootstrap_env"]
        or env["patch_bootstrap_env"]
        or env["vector_warmup_env"]
        or env["bootstrap_env"]
        or env["bootstrap_timers"]
    )
    warmup_requested = bool(
        warmup
        if warmup is not None
        else (
            env_requested
            or env["vector_warmup_env"]
            or _env_flag("VECTOR_METRICS_WARMUP", False)
            or env["bootstrap_timers"]
        )
    )
    resolved_bootstrap_fast = bool(
        bootstrap_fast
        if bootstrap_fast is not None
        else (
            env["vector_bootstrap_env"]
            or env["patch_bootstrap_env"]
            or env["bootstrap_env"]
            or env["bootstrap_timers"]
        )
    )
    resolved_bootstrap_fast = bool(
        resolved_bootstrap_fast
        or warmup_requested
        or env["bootstrap_env"]
        or env["bootstrap_timers"]
    )
    resolved_warmup = bool(warmup_requested and warmup is not False)

    return resolved_bootstrap_fast, resolved_warmup, env_requested, env["bootstrap_env"]


def get_shared_vector_metrics_db(
    *,
    bootstrap_fast: bool | None = None,
    warmup: bool | None = None,
    ensure_exists: bool | None = None,
    read_only: bool | None = None,
) -> "VectorMetricsDB":
    """Return a lazily initialised shared :class:`VectorMetricsDB` instance."""

    (
        resolved_bootstrap_fast,
        resolved_warmup,
        env_requested,
        bootstrap_env,
    ) = resolve_vector_bootstrap_flags(bootstrap_fast=bootstrap_fast, warmup=warmup)

    state_flags = _bootstrap_state_flags()
    menace_bootstrap = bool(_menace_bootstrap_active() or bootstrap_env)
    bootstrap_requested = bool(
        resolved_bootstrap_fast
        or resolved_warmup
        or env_requested
        or menace_bootstrap
        or _bootstrap_timers_active()
        or state_flags["bootstrap_state"]
        or state_flags["warmup_lite"]
    )
    timer_context = _bootstrap_timers_active()
    readiness_signal_active = bool(_READINESS_HOOK_ARMED)
    warmup_context_reasons = {
        "bootstrap_requested": bootstrap_requested,
        "bootstrap_timer": timer_context,
        "readiness_signal": readiness_signal_active,
        "warmup_env": _env_flag("VECTOR_METRICS_BOOTSTRAP_WARMUP", False),
        "menace_bootstrap": menace_bootstrap,
        "warmup_lite": state_flags["warmup_lite"],
        "bootstrap_state": state_flags["bootstrap_state"],
    }
    warmup_context = bool(any(warmup_context_reasons.values()))
    if warmup_context:
        resolved_warmup = True
        ensure_exists = False
        read_only = True
        _arm_shared_readiness_hook()

    global _VECTOR_DB_INSTANCE
    with _VECTOR_DB_LOCK:
        if _VECTOR_DB_INSTANCE is None:
            if warmup_context:
                _VECTOR_DB_INSTANCE = _BootstrapVectorMetricsStub(
                    path="vector_metrics.db",
                    bootstrap_fast=resolved_bootstrap_fast,
                    warmup=resolved_warmup,
                    ensure_exists=ensure_exists,
                    read_only=bool(read_only) if read_only is not None else False,
                )
                _VECTOR_DB_INSTANCE.activate_on_first_write()
                _VECTOR_DB_INSTANCE.register_readiness_hook()
                logger.info(
                    "vector_metrics_db.bootstrap.stub_selected",
                    extra={
                        "warmup_reasons": [
                            reason
                            for reason, active in warmup_context_reasons.items()
                            if active
                        ],
                        "bootstrap_fast": resolved_bootstrap_fast,
                        "warmup": resolved_warmup,
                        "read_only": bool(read_only) if read_only is not None else True,
                    },
                )
            else:
                _VECTOR_DB_INSTANCE = VectorMetricsDB(
                    "vector_metrics.db",
                    bootstrap_fast=resolved_bootstrap_fast,
                    warmup=resolved_warmup,
                    ensure_exists=ensure_exists,
                    read_only=bool(read_only) if read_only is not None else False,
                )
        elif isinstance(_VECTOR_DB_INSTANCE, _BootstrapVectorMetricsStub):
            _VECTOR_DB_INSTANCE.configure_activation(
                bootstrap_fast=resolved_bootstrap_fast,
                warmup=resolved_warmup,
                ensure_exists=False if warmup_context else ensure_exists,
                read_only=True if warmup_context else read_only,
            )
            _VECTOR_DB_INSTANCE.activate_on_first_write()
            _VECTOR_DB_INSTANCE.register_readiness_hook()
            if warmup_context:
                logger.info(
                    "vector_metrics_db.bootstrap.stub_reused",
                    extra={
                        "warmup_reasons": [
                            reason
                            for reason, active in warmup_context_reasons.items()
                            if active
                        ],
                        "bootstrap_fast": resolved_bootstrap_fast,
                        "warmup": resolved_warmup,
                        "read_only": bool(read_only) if read_only is not None else True,
                    },
                )

    if warmup_context:
        try:
            _VECTOR_DB_INSTANCE._log_deferred_activation(reason="bootstrap_warmup_summary")
        except Exception:
            logger.debug("vector metrics warmup summary logging failed", exc_info=True)
        logger.info(
            "vector_metrics_db.bootstrap.stub_returned",
            extra={
                "menace_bootstrap": menace_bootstrap,
                "bootstrap_requested": bootstrap_requested,
                "timer_context": timer_context,
                "warmup_env": _env_flag("VECTOR_METRICS_BOOTSTRAP_WARMUP", False),
            },
        )
        return _VECTOR_DB_INSTANCE

    _apply_pending_weights(_VECTOR_DB_INSTANCE)
    return _VECTOR_DB_INSTANCE


def activate_shared_vector_metrics_db(
    *, reason: str = "warmup_complete", post_warmup: bool = False
) -> "VectorMetricsDB | _BootstrapVectorMetricsStub":
    """Activate the shared vector metrics database after warmup."""

    (
        resolved_fast,
        resolved_warmup,
        env_requested,
        bootstrap_env,
    ) = resolve_vector_bootstrap_flags()

    bootstrap_context = bool(
        resolved_fast or resolved_warmup or env_requested or bootstrap_env
    )

    vm = get_shared_vector_metrics_db(
        bootstrap_fast=resolved_fast,
        warmup=resolved_warmup if bootstrap_context else False,
        ensure_exists=None if bootstrap_context else True,
        read_only=None if bootstrap_context else False,
    )

    if isinstance(vm, _BootstrapVectorMetricsStub):
        vm.register_readiness_hook()
        allow_activation = bool(post_warmup or reason == "bootstrap_ready")
        if not allow_activation:
            return vm
        vm.configure_activation(warmup=False, ensure_exists=True, read_only=False)
        delegate = vm._promote_from_stub(reason=reason or "warmup_complete")
        if delegate is not None:
            vm = delegate

    if isinstance(vm, VectorMetricsDB):
        vm.end_warmup(reason=reason)
        vm.activate_persistence(reason=reason)
    return vm


_MENACE_BOOTSTRAP_ENV_ACTIVE: bool | None = None


def _menace_bootstrap_active() -> bool:
    global _MENACE_BOOTSTRAP_ENV_ACTIVE
    if _MENACE_BOOTSTRAP_ENV_ACTIVE is None:
        _MENACE_BOOTSTRAP_ENV_ACTIVE = bool(_detect_bootstrap_environment()["bootstrap_env"])
    return _MENACE_BOOTSTRAP_ENV_ACTIVE


def get_bootstrap_vector_metrics_db(
    *,
    bootstrap_fast: bool | None = None,
    warmup: bool | None = None,
    ensure_exists: bool | None = None,
    read_only: bool | None = None,
) -> "VectorMetricsDB | _BootstrapVectorMetricsStub":
    """Return the shared DB with bootstrap-aware warmup defaults.

    The MENACE bootstrap signal (``MENACE_BOOTSTRAP*``) is latched on first
    access so callers that import this helper during bootstrap continue to use
    the warmup stub even if later environment mutations clear the flags.
    """

    menace_bootstrap = _menace_bootstrap_active()
    state_flags = _bootstrap_state_flags()
    (
        resolved_fast,
        warmup_mode,
        env_requested,
        bootstrap_context,
    ) = resolve_vector_bootstrap_flags(
        bootstrap_fast=bootstrap_fast, warmup=warmup
    )

    warmup_requested = bool(
        warmup_mode
        or env_requested
        or bootstrap_context
        or resolved_fast
        or menace_bootstrap
        or state_flags["bootstrap_state"]
        or state_flags["warmup_lite"]
    )
    if warmup_requested:
        warmup_mode = True
        ensure_exists = False
        read_only = True
        _arm_shared_readiness_hook()

    return get_shared_vector_metrics_db(
        bootstrap_fast=resolved_fast,
        warmup=warmup_mode,
        ensure_exists=ensure_exists,
        read_only=read_only,
    )


def ensure_vector_db_weights(
    db_names: Sequence[str],
    *,
    bootstrap_fast: bool | None = None,
    warmup: bool | None = None,
    ensure_exists: bool | None = None,
    read_only: bool | None = None,
) -> None:
    """Seed vector DB ranking weights without forcing SQLite initialisation."""

    names = [str(n) for n in db_names if n]
    if not names:
        return

    _record_pending_weights(names)

    if bootstrap_fast is None or warmup is None:
        bootstrap_fast, resolved_warmup, env_requested, bootstrap_env = (
            resolve_vector_bootstrap_flags(bootstrap_fast=bootstrap_fast, warmup=warmup)
        )
    else:
        resolved_warmup = bool(warmup)
        env_requested = False
        bootstrap_env = False

    if bootstrap_fast:
        logger.info(
            "vector_metrics_db.bootstrap.fast_weights_skipped",
            extra={
                "count": len(names),
                "menace_bootstrap": bootstrap_env,
                "env_bootstrap_requested": env_requested,
            },
        )

    try:
        vdb = get_shared_vector_metrics_db(
            bootstrap_fast=bootstrap_fast,
            warmup=resolved_warmup,
            ensure_exists=ensure_exists,
            read_only=read_only,
        )
        _apply_pending_weights(vdb)
        if resolved_warmup and not bootstrap_fast:
            logger.info(
                "vector_metrics_db.bootstrap.warmup_weights_cached",
                extra={"count": len(names)},
            )
    except Exception as exc:  # pragma: no cover - log only
        logger.warning("vector_metrics_db.bootstrap.weight_seed_failed", exc_info=exc)


class _StubCursor:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def fetchall(self):
        return []

    def fetchone(self):
        return None


class _StubConnection:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def execute(self, *args, **kwargs):
        sql = args[0] if args else ""
        self.logger.debug(
            "vector_metrics_db.stub.execute",
            extra={"sql": str(sql)},
        )
        return _StubCursor(self.logger)

    def executemany(self, *args, **kwargs):
        return self.execute(*args, **kwargs)

    def commit(self):
        self.logger.debug("vector_metrics_db.stub.commit")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


@dataclass
class VectorMetric:
    """Single vector operation metric record."""

    event_type: str
    db: str
    tokens: int = 0
    wall_time_ms: float = 0.0
    store_time_ms: float = 0.0
    hit: bool | None = None
    rank: int | None = None
    contribution: float | None = None
    prompt_tokens: int | None = None
    patch_id: str = ""
    session_id: str = ""
    vector_id: str = ""
    similarity: float | None = None
    context_score: float | None = None
    age: float | None = None
    win: bool | None = None
    regret: bool | None = None
    ts: str = datetime.utcnow().isoformat()


def default_vector_metrics_path(
    *,
    ensure_exists: bool | None = None,
    bootstrap_read_only: bool | None = None,
    read_only: bool = False,
) -> Path:
    """Return the default location for the vector metrics database.

    The original implementation relied on :func:`resolve_path` which raises a
    :class:`FileNotFoundError` when ``vector_metrics.db`` has not been created
    yet.  Some entry points instantiate :class:`VectorMetricsDB` (or import
    modules that resolve the default path) before the database exists, causing
    start-up to abort.  This helper mirrors the previous behaviour when the
    file is present while providing a deterministic fallback rooted at the
    repository when it is missing.

    When ``ensure_exists`` is ``True`` the parent directory is created and an
    empty file is touched so subsequent :func:`resolve_path` calls succeed once
    persistence is permitted.  ``sqlite3`` will initialise the actual database
    schema on first use.  When ``ensure_exists`` is ``False`` no filesystem
    writes are performed.  Passing ``read_only=True`` or enabling
    ``bootstrap_read_only`` forces ``ensure_exists`` to ``False`` unless the
    caller explicitly sets it, which helps avoid filesystem writes during
    ambiguous warmup and bootstrap flows.
    """

    if bootstrap_read_only is None:
        (
            resolved_bootstrap_fast,
            resolved_warmup,
            env_requested,
            bootstrap_context,
        ) = resolve_vector_bootstrap_flags()
        bootstrap_read_only = bool(
            resolved_bootstrap_fast or resolved_warmup or env_requested or bootstrap_context
        )

    if ensure_exists is None:
        ensure_exists = not (bootstrap_read_only or read_only)

    allow_fs_changes = bool(ensure_exists and not bootstrap_read_only and not read_only)
    path: Path | None = None

    if bootstrap_read_only:
        path = Path("vector_metrics.db").expanduser()
        if not path.is_absolute():
            path = path.resolve()
        return path

    if not bootstrap_read_only:
        try:
            path = _dynamic_path_router().resolve_path("vector_metrics.db")
        except FileNotFoundError:
            path = None

    if path is None:
        path = (_dynamic_path_router().get_project_root() / "vector_metrics.db").resolve()

    if allow_fs_changes:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.touch()

    return path


class VectorMetricsDB:
    """SQLite-backed store for :class:`VectorMetric` records.

    A strict warmup path is available via ``warmup=True`` (or the matching
    environment flags).  In this mode the database connection is replaced with
    a stub that never resolves the on-disk path or initialises the router
    until callers explicitly opt-in by invoking :meth:`activate_persistence`.
    The warmup stub is also auto-enabled whenever bootstrap environment flags
    (``MENACE_BOOTSTRAP*``, ``VECTOR_WARMUP``, etc.) are present so that
    bootstrap-time callers get a no-op connection without touching the
    filesystem.  ``read_only=True`` provides the same stubbed behaviour for
    callers that cannot confirm bootstrap status; ``ensure_exists`` can be set
    to ``False`` in those cases to prevent any filesystem writes.  Once
    bootstrap completes and real metrics need to be recorded, callers should
    call :meth:`activate_persistence` (or :meth:`activate_router`) to
    transition into the normal SQLite-backed flow.
    """

    def __init__(
        self,
        path: Path | str = "vector_metrics.db",
        *,
        bootstrap_safe: bool = False,
        bootstrap_fast: bool | None = None,
        warmup: bool | None = None,
        ensure_exists: bool | None = None,
        read_only: bool = False,
        allow_bootstrap_path_resolution: bool = False,
    ) -> None:
        bootstrap_safe = bootstrap_safe or _env_flag(
            "VECTOR_METRICS_BOOTSTRAP_SAFE", False
        )
        (
            resolved_bootstrap_fast,
            resolved_warmup,
            env_bootstrap,
            bootstrap_context,
        ) = resolve_vector_bootstrap_flags(
            bootstrap_fast=bootstrap_fast, warmup=warmup
        )
        self._bootstrap_timers_active = _bootstrap_timers_active()
        self._menace_bootstrap_env = bootstrap_context
        self._bootstrap_env_requested = env_bootstrap
        self._bootstrap_context = bool(
            resolved_bootstrap_fast
            or resolved_warmup
            or env_bootstrap
            or self._bootstrap_timers_active
        )
        self._router_init_blocked = bool(
            (self._bootstrap_context or self._bootstrap_timers_active)
            and not allow_bootstrap_path_resolution
        )
        if self._bootstrap_context and warmup is None:
            resolved_warmup = True
        if self._bootstrap_timers_active and warmup is None:
            resolved_warmup = True
        explicit_bootstrap_stub_opt_out = bool(warmup is False and not resolved_bootstrap_fast)
        self._warmup_override_disabled = bool(explicit_bootstrap_stub_opt_out)
        self.bootstrap_fast = resolved_bootstrap_fast
        self._warmup_mode = (
            False if self._warmup_override_disabled else resolved_warmup
        )
        self._read_only = bool(read_only)
        self._env_warmup_opt_in = bool(
            self._bootstrap_context
            or env_bootstrap
            or resolved_warmup
            or resolved_bootstrap_fast
            or self._bootstrap_timers_active
            or self._router_init_blocked
        )
        self._lazy_mode = True
        self._boot_stub_active = bool(
            self._read_only
            or (
                self._bootstrap_context
                or self.bootstrap_fast
                or self._warmup_mode
                or self._bootstrap_timers_active
                or self._env_warmup_opt_in
                or self._router_init_blocked
            )
            and not explicit_bootstrap_stub_opt_out
        )
        if self._warmup_mode and not self._boot_stub_active:
            self._boot_stub_active = True
            logger.info(
                "vector_metrics_db.bootstrap.warmup_stub_forced",
                extra=_timestamp_payload(
                    None,
                    configured_path=str(path),
                    warmup_mode=self._warmup_mode,
                    menace_bootstrap=self._bootstrap_context,
                    env_bootstrap_requested=self._env_warmup_opt_in,
                ),
            )
        self._lazy_primed = self._boot_stub_active
        if ensure_exists is None:
            self._default_ensure_exists = not (
                self._boot_stub_active
                or self.bootstrap_fast
                or self._warmup_mode
                or self._read_only
            )
        else:
            self._default_ensure_exists = bool(ensure_exists and not self._read_only)
        self._commit_required = False
        self._commit_reason = "first_use"
        self._stub_conn = _StubConnection(logger)
        self._stub_buffer: list[VectorMetric] = []
        self._stub_buffer_limit = 256
        self._stub_overflow_logged = False
        self._readiness_hook_registered = False
        self._pending_readiness_hook = bool(self._boot_stub_active)
        self._ready_probe_logged = False
        self._persistence_ready_logged = False
        self._stub_usage_logged = False
        self._persistence_activated = not self._boot_stub_active
        self._persistence_activation_pending = bool(self._boot_stub_active)
        self._activate_on_first_write = False
        self._prometheus_ready = False
        self._warmup_complete = threading.Event()
        if not self._warmup_mode:
            self._warmup_complete.set()
        self._bootstrap_guard_active = bool(
            self._boot_stub_active
            and (self._warmup_mode or self.bootstrap_fast or self._bootstrap_context)
        )
        if self._bootstrap_guard_active:
            self._default_ensure_exists = False
            logger.info(
                "vector_metrics_db.bootstrap.stub_guard",
                extra=_timestamp_payload(
                    None,
                    configured_path=str(path),
                    warmup_mode=self._warmup_mode,
                    bootstrap_fast=self.bootstrap_fast,
                    menace_bootstrap=self._bootstrap_context,
                ),
            )
        init_start = time.perf_counter()
        if not self._warmup_mode:
            logger.info(
                "vector_metrics_db.init.start",
                extra=_timestamp_payload(
                    init_start,
                    configured_path=str(path),
                    lazy_mode=self._lazy_mode,
                    bootstrap_fast=self.bootstrap_fast,
                    warmup_mode=self._warmup_mode,
                    stub_mode=self._boot_stub_active,
                ),
            )
        elif self._boot_stub_active:
            logger.info(
                "vector_metrics_db.init.stub",
                extra=_timestamp_payload(
                    init_start,
                    configured_path=str(path),
                    lazy_mode=self._lazy_mode,
                    bootstrap_fast=self.bootstrap_fast,
                    warmup_mode=self._warmup_mode,
                    stub_mode=True,
                    menace_bootstrap=bootstrap_context,
                    env_bootstrap_requested=env_bootstrap,
                ),
            )
        if self._boot_stub_active and self._env_warmup_opt_in:
            logger.info(
                "vector_metrics_db.bootstrap.stub_env_opt_in",
                extra=_timestamp_payload(
                    init_start,
                    menace_bootstrap=bootstrap_context,
                    env_bootstrap_requested=env_bootstrap,
                    warmup_mode=self._warmup_mode,
                    bootstrap_fast=self.bootstrap_fast,
                ),
            )
        if self._boot_stub_active and self._bootstrap_context:
            logger.info(
                "vector_metrics_db.bootstrap.stub_armed",
                extra=_timestamp_payload(
                    init_start,
                    menace_bootstrap=True,
                    activation_hook="activate_persistence",
                    configured_path=str(path),
                    warmup_mode=self._warmup_mode,
                    bootstrap_fast=self.bootstrap_fast,
                ),
            )

        self._cached_weights: dict[str, float] = {}
        self._schema_cache: dict[str, list[str]] = {}
        self._default_columns: dict[str, list[str]] = {}
        self._schema_defaults_initialized = False
        self._bootstrap_safe = bootstrap_safe
        self._configured_path = Path(path)
        self._resolved_path: Path | None = None
        self._default_path: Path | None = None
        self.router = None
        self._conn = None

        if self._boot_stub_active or self._warmup_mode:
            logger.info(
                "vector_metrics_db.bootstrap.stub_path_active",
                extra=_timestamp_payload(
                    init_start,
                    configured_path=str(path),
                    menace_bootstrap=self._bootstrap_context,
                    bootstrap_deadlines=self._bootstrap_timers_active,
                    warmup_mode=self._warmup_mode,
                ),
            )
            logger.info(
                "vector_metrics_db.bootstrap.stubbed_persistence_deferred",
                extra=_timestamp_payload(
                    init_start,
                    configured_path=str(path),
                    warmup_mode=self._warmup_mode,
                    bootstrap_fast=self.bootstrap_fast,
                    menace_bootstrap=self._bootstrap_context,
                ),
            )
            if self._pending_readiness_hook:
                self._register_readiness_hook()
            self._conn = self._stub_conn
            return

        eager_resolve = bool(ensure_exists)
        if eager_resolve:
            self._resolved_path, self._default_path = self._resolve_requested_path(
                self._configured_path, ensure_exists=eager_resolve
            )

    def _initialize_schema_defaults(self) -> None:
        if self._schema_defaults_initialized:
            return
        self._schema_defaults_initialized = True
        self._schema_cache = {}
        self._default_columns = {
            "vector_metrics": [
                "event_type",
                "db",
                "tokens",
                "wall_time_ms",
                "store_time_ms",
                "hit",
                "rank",
                "contribution",
                "prompt_tokens",
                "patch_id",
                "session_id",
                "vector_id",
                "similarity",
                "context_score",
                "age",
                "win",
                "regret",
                "ts",
            ],
            "patch_ancestry": [
                "patch_id",
                "vector_id",
                "rank",
                "contribution",
                "license",
                "semantic_alerts",
                "alignment_severity",
                "risk_score",
            ],
            "patch_metrics": [
                "patch_id",
                "errors",
                "tests_passed",
                "lines_changed",
                "context_tokens",
                "patch_difficulty",
                "start_time",
                "time_to_completion",
                "error_trace_count",
                "roi_tag",
                "effort_estimate",
                "enhancement_score",
            ],
        }

    def _register_readiness_hook(self) -> None:
        if self._readiness_hook_registered:
            return
        if not self._boot_stub_active:
            return
        try:
            from bootstrap_readiness import readiness_signal
        except Exception:  # pragma: no cover - optional dependency
            logger.debug("vector metrics readiness hook unavailable", exc_info=True)
            return

        self._readiness_hook_registered = True

        def _await_bootstrap_ready() -> None:
            try:
                readiness_signal().await_ready(timeout=None)
            except Exception:  # pragma: no cover - best effort logging
                logger.debug(
                    "vector metrics readiness gate failed", exc_info=True
                )
                return
            try:
                self.activate_on_first_write()
                self._mark_warmup_complete(
                    reason="bootstrap_ready", activate=False
                )
                if self._warmup_mode and not self._warmup_complete.is_set():
                    logger.info(
                        "vector_metrics_db.bootstrap.persistence_deferred",
                        extra=_timestamp_payload(
                            None,
                            warmup_mode=self._warmup_mode,
                            bootstrap_fast=self.bootstrap_fast,
                            menace_bootstrap=self._bootstrap_context,
                            reason="warmup_in_progress",
                        ),
                    )
                    self._warmup_complete.wait()
            except Exception:  # pragma: no cover - best effort logging
                logger.debug(
                    "vector metrics readiness gate handling failed", exc_info=True
                )

        threading.Thread(
            target=_await_bootstrap_ready,
            name="vector-metrics-readiness",
            daemon=True,
        ).start()

    def _buffer_stub_metric(self, rec: VectorMetric) -> None:
        if len(self._stub_buffer) < self._stub_buffer_limit:
            self._stub_buffer.append(rec)
            return
        if self._stub_overflow_logged:
            return
        self._stub_overflow_logged = True
        logger.info(
            "vector_metrics_db.bootstrap.stub_buffer_overflow",
            extra={
                "buffer_limit": self._stub_buffer_limit,
                "bootstrap_fast": self.bootstrap_fast,
                "warmup_mode": self._warmup_mode,
            },
        )

    def _flush_stub_buffer(self) -> None:
        if not self._stub_buffer:
            return
        buffered = list(self._stub_buffer)
        self._stub_buffer.clear()
        logger.info(
            "vector_metrics_db.bootstrap.flush_stub_buffer",
            extra=_timestamp_payload(None, count=len(buffered)),
        )
        for rec in buffered:
            try:
                self.add(rec)
            except Exception:  # pragma: no cover - best effort flush
                logger.debug("vector_metrics_db.bootstrap.flush_failed", exc_info=True)

    def _exit_lazy_mode(self, *, reason: str) -> None:
        """Upgrade from bootstrap stub to full schema on first meaningful use."""

        init_start = time.perf_counter()
        logger.info(
            "vector_metrics_db.bootstrap.lazy_exit.start",
            extra=_timestamp_payload(
                init_start,
                reason=reason,
                warmup_mode=self._warmup_mode,
                bootstrap_fast=self.bootstrap_fast,
                stub_mode=self._boot_stub_active,
            ),
        )
        if self._warmup_mode or self.bootstrap_fast:
            self._mark_warmup_complete(reason=reason)
        self._read_only = False
        self._boot_stub_active = False
        self._lazy_mode = False
        self._lazy_primed = False
        self._persistence_activated = True
        self._persistence_activation_pending = False
        self._initialize_schema_defaults()
        self._prepare_connection(init_start)
        self._flush_stub_buffer()
        self._commit_required = False
        self._commit_reason = "first_use"
        logger.info(
            "vector_metrics_db.bootstrap.lazy_exit.complete",
            extra=_timestamp_payload(init_start, resolved_path=str(self._resolved_path)),
        )

    def _mark_warmup_complete(self, *, reason: str, activate: bool = True) -> None:
        if self._warmup_complete.is_set():
            return
        self._warmup_complete.set()
        if self._warmup_mode or self.bootstrap_fast:
            logger.info(
                "vector_metrics_db.bootstrap.warmup_complete",
                extra=_timestamp_payload(
                    None,
                    warmup_mode=self._warmup_mode,
                    bootstrap_fast=self.bootstrap_fast,
                    reason=reason,
                ),
            )
        self._warmup_mode = False
        self.bootstrap_fast = False

        if self._persistence_activation_pending and activate:
            logger.info(
                "vector_metrics_db.bootstrap.persistence_retry",
                extra=_timestamp_payload(
                    None,
                    warmup_mode=self._warmup_mode,
                    bootstrap_fast=self.bootstrap_fast,
                    stub_mode=self._boot_stub_active,
                    reason=reason,
                ),
            )
            self.activate_persistence(reason=reason)

    def end_warmup(
        self, *, reason: str = "warmup_complete", activate: bool = True
    ) -> None:
        """Mark warmup as complete and trigger pending activation."""

        if self._warmup_complete.is_set():
            return
        self._mark_warmup_complete(reason=reason, activate=activate)

    def _should_skip_logging(self) -> bool:
        if not self._boot_stub_active:
            return False
        if not _noop_logging(self.bootstrap_fast, self._warmup_mode):
            return False
        if self._lazy_primed:
            start = time.perf_counter()
            logger.info(
                "vector_metrics_db.bootstrap.noop_logging",
                extra=_timestamp_payload(
                    start,
                    warmup_mode=self._warmup_mode,
                    bootstrap_fast=self.bootstrap_fast,
                    configured_path=str(self._configured_path),
                    activation_hook="activate_persistence",
                    menace_bootstrap=self._bootstrap_context,
                ),
            )
            self._lazy_primed = False
        return True

    def activate_persistence(self, *, reason: str = "metrics_ready") -> None:
        """Exit warmup/bootstrap mode and initialise SQLite lazily."""

        if not self._boot_stub_active:
            self._persistence_activation_pending = False
            return
        if self._bootstrap_guard_active and not self._pending_readiness_hook:
            self._pending_readiness_hook = True
        if self._pending_readiness_hook and not self._readiness_hook_registered:
            self._register_readiness_hook()
        warmup_guard_active = (
            (self._warmup_mode or self.bootstrap_fast or self._bootstrap_context)
            and not self._warmup_complete.is_set()
        )
        if warmup_guard_active and reason != "first_write":
            self._persistence_activation_pending = True
            logger.info(
                "vector_metrics_db.bootstrap.persistence_pending",
                extra=_timestamp_payload(
                    None,
                    warmup_mode=self._warmup_mode,
                    bootstrap_fast=self.bootstrap_fast,
                    stub_mode=self._boot_stub_active,
                    menace_bootstrap=self._bootstrap_context,
                    reason=reason,
                ),
            )
            return
        logger.info(
            "vector_metrics_db.bootstrap.activate_persistence",
            extra=_timestamp_payload(
                None,
                warmup_mode=self._warmup_mode,
                bootstrap_fast=self.bootstrap_fast,
                reason=reason,
                stub_mode=self._boot_stub_active,
            ),
        )
        self._persistence_activation_pending = False
        self._exit_lazy_mode(reason=reason)
        _apply_pending_weights(self)
        logger.info(
            "vector_metrics_db.bootstrap.persistence_activated",
            extra=_timestamp_payload(
                None,
                resolved_path=str(self._resolved_path) if self._resolved_path else None,
                default_path=str(self._default_path) if self._default_path else None,
                stub_mode=self._boot_stub_active,
                warmup_mode=self._warmup_mode,
            ),
        )
        if not self._persistence_ready_logged:
            self._persistence_ready_logged = True
            logger.info(
                "vector_metrics_db.bootstrap.persistence_ready",
                extra=_timestamp_payload(
                    None,
                    resolved_path=str(self._resolved_path),
                    reason=reason,
                ),
            )

    def activate_router(self, *, reason: str = "bootstrap_complete") -> "VectorMetricsDB":
        """Convert the bootstrap stub into a real router-backed instance."""

        self.activate_persistence(reason=reason)
        _ = self.conn
        return self

    def activate_on_first_write(self) -> None:
        """Defer persistence activation until the first metric is recorded."""

        self._activate_on_first_write = True

    def planned_path(self) -> Path:
        """Return the resolved database path without touching the filesystem."""

        if self._boot_stub_active and self._resolved_path is None:
            return Path(self._configured_path).expanduser()
        if self._resolved_path is None or self._default_path is None:
            self._resolved_path, self._default_path = self._resolve_requested_path(
                self._configured_path, ensure_exists=False
            )
        return self._resolved_path

    @property
    def conn(self):
        return self._connection(
            reason=self._commit_reason,
            commit_required=self._commit_required,
        )

    @conn.setter
    def conn(self, value):
        self._conn = value

    def _connection(self, *, reason: str, commit_required: bool):
        if commit_required:
            self._commit_required = True
            self._commit_reason = reason
        if self._conn is not None:
            return self._conn
        if self._lazy_mode:
            if self._boot_stub_active:
                return self._stub_conn
            self._exit_lazy_mode(reason=reason)
            return self._conn
        return self._conn or self._stub_conn

    def _conn_for(self, *, reason: str, commit_required: bool = True):
        if self._boot_stub_active:
            if not self._stub_usage_logged:
                self._stub_usage_logged = True
                logger.info(
                    "vector_metrics_db.bootstrap.stub_connection",
                    extra={
                        "reason": reason,
                        "warmup_mode": self._warmup_mode,
                        "bootstrap_fast": self.bootstrap_fast,
                        "menace_bootstrap": self._menace_bootstrap_env,
                        "env_bootstrap_requested": self._bootstrap_env_requested,
                    },
                )
            commit_required = False
        conn = self._connection(reason=reason, commit_required=commit_required)
        if commit_required:
            self._commit_required = False
            self._commit_reason = "first_use"
        return conn

    @classmethod
    def warmup_path_probe(cls, path: Path | str = "vector_metrics.db") -> str:
        """Report the configured path without touching the filesystem."""

        return str(Path(path).expanduser())

    def readiness_probe(self) -> dict[str, Any]:
        """Log and return bootstrap readiness without touching disk."""

        status = {
            "configured_path": str(Path(self._configured_path).expanduser()),
            "stub_mode": self._boot_stub_active,
            "warmup_mode": self._warmup_mode,
            "bootstrap_fast": self.bootstrap_fast,
            "bootstrap_deadlines": self._bootstrap_timers_active,
            "env_bootstrap_requested": self._bootstrap_env_requested,
            "menace_bootstrap": self._menace_bootstrap_env,
            "persistence_pending": self._persistence_activation_pending,
        }
        if not self._ready_probe_logged:
            self._ready_probe_logged = True
            logger.info(
                "vector_metrics_db.bootstrap.readiness_probe",
                extra=_timestamp_payload(None, **status),
            )
        return status

    def ready_probe(self) -> str:
        """Return the resolved database path without any I/O."""
        status = self.readiness_probe()
        if self._boot_stub_active and self._resolved_path is None:
            return status["configured_path"]
        if self._resolved_path is None or self._default_path is None:
            self._resolved_path, self._default_path = self._resolve_requested_path(
                self._configured_path,
                ensure_exists=False,
            )
        return str(self._resolved_path)

    def persistence_probe(self) -> bool:
        """Indicate whether persistence has been activated without any I/O."""

        return bool(self._persistence_activated and not self._boot_stub_active)

    def _resolve_requested_path(
        self, path: Path, *, ensure_exists: bool
    ) -> tuple[Path, Path]:
        if self._boot_stub_active or self._warmup_mode:
            requested = Path(path).expanduser()
            return requested, requested
        default_path = default_vector_metrics_path(
            ensure_exists=ensure_exists,
            bootstrap_read_only=self._boot_stub_active,
            read_only=self._read_only,
        )
        requested = Path(path).expanduser()
        if str(requested.as_posix()) == "vector_metrics.db":
            p = default_path
        else:
            if not requested.is_absolute():
                requested = (default_path.parent / requested).resolve()
            else:
                requested = requested.resolve()
            if ensure_exists:
                requested.parent.mkdir(parents=True, exist_ok=True)
            p = requested
        return p, default_path

    def _prepare_connection(self, init_start: float | None = None) -> None:
        init_start = init_start or time.perf_counter()
        if self._conn is not None:
            return
        if self._warmup_mode or self._boot_stub_active:
            logger.info(
                "vector_metrics_db.bootstrap.fast_return",
                extra=_timestamp_payload(
                    init_start,
                    warmup_mode=self._warmup_mode,
                    bootstrap_fast=self.bootstrap_fast,
                    stub_mode=self._boot_stub_active,
                ),
            )
            return
        if not self._persistence_activated:
            logger.info(
                "vector_metrics_db.bootstrap.persistence_suspended",
                extra=_timestamp_payload(
                    init_start,
                    warmup_mode=self._warmup_mode,
                    bootstrap_fast=self.bootstrap_fast,
                    stub_mode=self._boot_stub_active,
                ),
            )
            return
        if self._bootstrap_context and not self._warmup_mode:
            logger.info(
                "vector_metrics_db.bootstrap.eager_initialization",
                extra=_timestamp_payload(
                    init_start,
                    bootstrap_fast=self.bootstrap_fast,
                    warmup_mode=self._warmup_mode,
                    configured_path=str(self._configured_path),
                ),
            )

        logger.info(
            "vector_metrics_db.initialization.mode",
            extra=_timestamp_payload(
                init_start,
                bootstrap_fast=self.bootstrap_fast,
                warmup_mode=self._warmup_mode,
                lazy_primed=self._lazy_primed,
            ),
        )

        router_mod = _db_router()

        router_mod.LOCAL_TABLES.add("vector_metrics")

        if self._bootstrap_safe:
            router_mod.set_audit_bootstrap_safe_default(True)

        if self._resolved_path is None or self._default_path is None:
            self._resolved_path, self._default_path = self._resolve_requested_path(
                self._configured_path, ensure_exists=True
            )
        else:
            _ = self._resolve_requested_path(  # ensure directory exists lazily
                self._configured_path, ensure_exists=True
            )

        logger.info(
            "vector_metrics_db.path.resolved",
            extra=_timestamp_payload(
                init_start,
                resolved_path=str(self._resolved_path),
                default_path=str(self._default_path),
            ),
        )

        if (
            router_mod.GLOBAL_ROUTER is not None
            and self._resolved_path == self._default_path
        ):
            self.router = router_mod.GLOBAL_ROUTER
            using_global_router = True
            if self._bootstrap_safe:
                self.router.local_conn.audit_bootstrap_safe = True
                self.router.shared_conn.audit_bootstrap_safe = True
        else:
            self.router = router_mod.init_db_router(
                "vector_metrics_db",
                str(self._resolved_path),
                str(self._resolved_path),
                bootstrap_safe=self._bootstrap_safe,
            )
            using_global_router = False
        wal = self._resolved_path.with_suffix(self._resolved_path.suffix + "-wal")
        shm = self._resolved_path.with_suffix(self._resolved_path.suffix + "-shm")
        for sidecar in (wal, shm):
            try:
                if sidecar.exists():
                    logger.warning(
                        "vector_metrics_db.sidecar.present",
                        extra=_timestamp_payload(
                            init_start,
                            sidecar=str(sidecar),
                            size=sidecar.stat().st_size,
                        ),
                    )
            except Exception:  # pragma: no cover - best effort diagnostics
                logger.exception("vector_metrics_db.sidecar.inspect_failed")

        self._conn = self.router.get_connection("vector_metrics")
        logger.info(
            "vector_metrics_db.connection.ready",
            extra=_timestamp_payload(
                init_start, using_global_router=using_global_router
            ),
        )
        schema_start = time.perf_counter()
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vector_metrics(
                event_type TEXT,
                db TEXT,
                tokens INTEGER,
                wall_time_ms REAL,
                store_time_ms REAL,
                hit INTEGER,
                rank INTEGER,
                contribution REAL,
                prompt_tokens INTEGER,
                patch_id TEXT,
                session_id TEXT,
                vector_id TEXT,
                similarity REAL,
                context_score REAL,
                age REAL,
                win INTEGER,
                regret INTEGER,
                ts TEXT
            )
            """
        )
        logger.info(
            "vector_metrics_db.schema.ensured",
            extra=_timestamp_payload(schema_start),
        )
        migration_start = time.perf_counter()
        if not self.bootstrap_fast:
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS vector_metrics_event_db_ts
                    ON vector_metrics(event_type, db, ts)
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS patch_ancestry(
                    patch_id TEXT,
                    vector_id TEXT,
                    rank INTEGER,
                    contribution REAL,
                    license TEXT,
                    semantic_alerts TEXT,
                    alignment_severity REAL,
                    risk_score REAL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS patch_metrics(
                    patch_id TEXT PRIMARY KEY,
                    errors TEXT,
                    tests_passed INTEGER,
                    lines_changed INTEGER,
                    context_tokens INTEGER,
                    patch_difficulty INTEGER,
                    start_time REAL,
                    time_to_completion REAL,
                    error_trace_count INTEGER,
                    roi_tag TEXT,
                    effort_estimate REAL,
                    enhancement_score REAL
                )
                """
            )
            # Store adaptive ranking weights so the ranker can learn over time.
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ranking_weights(
                    db TEXT PRIMARY KEY,
                    weight REAL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vector_weights(
                    vector_id TEXT PRIMARY KEY,
                    weight REAL
                )
                """
            )
            # Persist session vector data so retrievals can be reconciled after
            # restarts.  Stored as JSON blobs keyed by ``session_id``.
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pending_sessions(
                    session_id TEXT PRIMARY KEY,
                    vectors TEXT,
                    metadata TEXT
                )
                """
            )
            self._conn.commit()
            cols = self._table_columns("vector_metrics")
            migrations = {
                "session_id": "ALTER TABLE vector_metrics ADD COLUMN session_id TEXT",
                "vector_id": "ALTER TABLE vector_metrics ADD COLUMN vector_id TEXT",
                "similarity": "ALTER TABLE vector_metrics ADD COLUMN similarity REAL",
                "context_score": "ALTER TABLE vector_metrics ADD COLUMN context_score REAL",
                "age": "ALTER TABLE vector_metrics ADD COLUMN age REAL",
                "win": "ALTER TABLE vector_metrics ADD COLUMN win INTEGER",
                "regret": "ALTER TABLE vector_metrics ADD COLUMN regret INTEGER",
            }
            applied_columns = []
            for name, stmt in migrations.items():
                if name not in cols:
                    self._conn.execute(stmt)
                    applied_columns.append(name)
            logger.info(
                "vector_metrics_db.migrations.vector_metrics",
                extra=_timestamp_payload(
                    migration_start, applied_columns=applied_columns
                ),
            )
            self._conn.commit()
            pcols = self._table_columns("patch_ancestry")
            if "license" not in pcols:
                self._conn.execute("ALTER TABLE patch_ancestry ADD COLUMN license TEXT")
            if "semantic_alerts" not in pcols:
                self._conn.execute(
                    "ALTER TABLE patch_ancestry ADD COLUMN semantic_alerts TEXT"
                )
            if "alignment_severity" not in pcols:
                self._conn.execute(
                    "ALTER TABLE patch_ancestry ADD COLUMN alignment_severity REAL"
                )
            if "risk_score" not in pcols:
                self._conn.execute(
                    "ALTER TABLE patch_ancestry ADD COLUMN risk_score REAL"
                )
            self._conn.commit()
            mcols = self._table_columns("patch_metrics")
            if "context_tokens" not in mcols:
                self._conn.execute(
                    "ALTER TABLE patch_metrics ADD COLUMN context_tokens INTEGER"
                )
            if "patch_difficulty" not in mcols:
                self._conn.execute(
                    "ALTER TABLE patch_metrics ADD COLUMN patch_difficulty INTEGER"
                )
            if "start_time" not in mcols:
                self._conn.execute(
                    "ALTER TABLE patch_metrics ADD COLUMN start_time REAL"
                )
            if "time_to_completion" not in mcols:
                self._conn.execute(
                    "ALTER TABLE patch_metrics ADD COLUMN time_to_completion REAL"
                )
            if "error_trace_count" not in mcols:
                self._conn.execute(
                    "ALTER TABLE patch_metrics ADD COLUMN error_trace_count INTEGER"
                )
            if "roi_tag" not in mcols:
                self._conn.execute(
                    "ALTER TABLE patch_metrics ADD COLUMN roi_tag TEXT"
                )
            if "effort_estimate" not in mcols:
                self._conn.execute(
                    "ALTER TABLE patch_metrics ADD COLUMN effort_estimate REAL"
                )
            if "enhancement_score" not in mcols:
                self._conn.execute(
                    "ALTER TABLE patch_metrics ADD COLUMN enhancement_score REAL"
                )
            self._conn.commit()
            logger.info(
                "vector_metrics_db.migrations.patch_tables",
                extra=_timestamp_payload(
                    migration_start,
                    patch_ancestry_missing=[
                        c
                        for c in (
                            "license",
                            "semantic_alerts",
                            "alignment_severity",
                            "risk_score",
                        )
                        if c not in pcols
                    ],
                    patch_metrics_missing=[
                        c
                        for c in (
                            "context_tokens",
                            "patch_difficulty",
                            "start_time",
                            "time_to_completion",
                            "error_trace_count",
                            "roi_tag",
                            "effort_estimate",
                            "enhancement_score",
                        )
                        if c not in mcols
                    ],
                ),
            )
        else:
            logger.info(
                "vector_metrics_db.bootstrap.fast_path_enabled",
                extra=_timestamp_payload(migration_start),
            )
            self._schema_cache.update(self._default_columns)

        self._lazy_primed = False
        self._lazy_mode = False
        logger.info(
            "vector_metrics_db.init.complete",
            extra=_timestamp_payload(
                init_start,
                resolved_path=str(self._resolved_path),
                using_global_router=using_global_router,
            ),
        )

    def _ensure_prometheus_ready(self) -> None:
        if self._prometheus_ready:
            return
        if self._boot_stub_active or self._warmup_mode:
            return
        try:
            _ensure_prometheus_objects()
            self._prometheus_ready = True
        except Exception:  # pragma: no cover - best effort registration
            logger.debug("vector_metrics_db.prometheus.init_failed", exc_info=True)

    def _table_columns(self, table: str) -> list[str]:
        """Return column names for ``table`` using non-blocking pragmas."""

        start = time.perf_counter()
        if self._boot_stub_active:
            return list(self._default_columns.get(table, []))
        self._initialize_schema_defaults()
        if self.bootstrap_fast:
            columns = self._schema_cache.get(table) or self._default_columns.get(
                table, []
            )
            logger.info(
                "vector_metrics_db.schema.cached",
                extra=_timestamp_payload(
                    start,
                    table=table,
                    column_count=len(columns),
                    fast_path=True,
                ),
            )
            return list(columns)
        timeout_ms: int | None = None
        conn = self._conn_for(reason="schema.inspect")
        try:
            if self.bootstrap_fast:
                try:
                    timeout_ms = int(conn.execute("PRAGMA busy_timeout").fetchone()[0])
                    conn.execute("PRAGMA busy_timeout = 0")
                except Exception:
                    logger.debug(
                        "vector_metrics_db.schema.fast_timeout_unset",
                        exc_info=True,
                    )
            rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        except Exception as exc:
            cache = self._schema_cache.get(table) or self._default_columns.get(table, [])
            logger.warning(
                "vector_metrics_db.schema.fast_path",
                extra=_timestamp_payload(
                    start,
                    table=table,
                    cached=len(cache),
                    reason=str(exc),
                ),
            )
            return list(cache)
        finally:
            if timeout_ms is not None:
                try:
                    conn.execute(f"PRAGMA busy_timeout = {timeout_ms}")
                except Exception:
                    logger.debug(
                        "vector_metrics_db.schema.restore_timeout_failed",
                        exc_info=True,
                    )
        logger.info(
            "vector_metrics_db.schema.inspected",
            extra=_timestamp_payload(start, table=table, column_count=len(rows)),
        )
        columns = [r[1] for r in rows]
        self._schema_cache[table] = columns
        return columns

    # ------------------------------------------------------------------
    def get_db_weights(
        self,
        *,
        bootstrap: bool = False,
        default: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        """Return mapping of origin database to current ranking weight."""

        pending_weights = _pending_weight_mapping()
        default_weights = dict(default or self._cached_weights or pending_weights)
        if _noop_logging(self.bootstrap_fast, self._warmup_mode):
            return default_weights

        if self._boot_stub_active:
            return default_weights

        if self.router is None:
            _ = self._conn_for(reason="get_db_weights.router")
        if self.router is None:
            return default_weights

        timeout_ms = (
            self.router.bootstrap_timeout_ms
            if (bootstrap or self.bootstrap_fast)
            else None
        )
        start = time.perf_counter()
        connection_ctx: contextlib.AbstractContextManager = (
            self.router.bootstrap_connection("ranking_weights", timeout_ms=timeout_ms)
            if timeout_ms is not None
            else contextlib.nullcontext(
                self._conn_for(reason="get_db_weights.read")
            )
        )
        try:
            with connection_ctx as conn:
                cur = conn.execute("SELECT db, weight FROM ranking_weights")
                rows = cur.fetchall()
        except Exception as exc:  # pragma: no cover - defensive bootstrap fallback
            logger.info(
                "vector_metrics_db.weights.cached",
                extra=_timestamp_payload(
                    start, cached=len(default_weights), reason=str(exc)
                ),
            )
            return default_weights
        weights = {str(db): float(weight) for db, weight in rows}
        self._cached_weights = dict(weights)
        if pending_weights:
            _clear_pending_weights(weights.keys())
        return weights

    # ------------------------------------------------------------------
    def update_db_weight(self, db: str, delta: float, *, normalize: bool = False) -> float:
        """Adjust ranking weight for *db* by ``delta`` and persist it.

        Weights are clamped to the inclusive range ``[0, 1]`` so repeated
        positive or negative feedback cannot push them outside sensible
        bounds.  When ``normalize`` is ``True`` all weights are also
        renormalised so their sum equals 1.0.  The new weight for ``db`` is
        returned (after optional normalisation)."""

        conn = self._conn_for(reason="update_db_weight")
        cur = conn.execute("SELECT weight FROM ranking_weights WHERE db=?", (db,))
        row = cur.fetchone()
        weight = float(row[0]) if row and row[0] is not None else 0.0
        weight = max(0.0, min(1.0, weight + delta))
        conn.execute(
            "REPLACE INTO ranking_weights(db, weight) VALUES(?, ?)", (db, weight)
        )
        conn.commit()
        if normalize:
            return self.normalize_db_weights().get(db, weight)
        return weight

    # ------------------------------------------------------------------
    def normalize_db_weights(self) -> dict[str, float]:
        """Scale all weights so they sum to 1.0.

        Returns the normalised weight mapping."""

        conn = self._conn_for(reason="normalize_db_weights")
        cur = conn.execute("SELECT db, weight FROM ranking_weights")
        rows = [(str(db), float(w)) for db, w in cur.fetchall()]
        total = sum(w for _db, w in rows)
        if total > 0:
            for db, w in rows:
                norm = w / total
                conn.execute(
                    "REPLACE INTO ranking_weights(db, weight) VALUES(?, ?)",
                    (db, norm),
                )
            conn.commit()
            return {db: w / total for db, w in rows}
        return {db: w for db, w in rows}

    # ------------------------------------------------------------------
    def set_db_weights(self, weights: Dict[str, float]) -> None:
        """Persist full ranking weight mapping."""

        rows = [
            (str(db), max(0.0, min(1.0, float(w)))) for db, w in weights.items()
        ]
        conn = self._conn_for(reason="set_db_weights")
        conn.executemany(
            "REPLACE INTO ranking_weights(db, weight) VALUES(?, ?)", rows
        )
        conn.commit()

    # ------------------------------------------------------------------
    def update_vector_weight(self, vector_id: str, delta: float) -> float:
        """Adjust ranking weight for *vector_id* by ``delta`` and persist it."""

        conn = self._conn_for(reason="update_vector_weight")
        cur = conn.execute("SELECT weight FROM vector_weights WHERE vector_id=?", (vector_id,))
        row = cur.fetchone()
        weight = float(row[0]) if row and row[0] is not None else 0.0
        weight = max(0.0, min(1.0, weight + delta))
        conn.execute(
            "REPLACE INTO vector_weights(vector_id, weight) VALUES(?, ?)",
            (vector_id, weight),
        )
        conn.commit()
        return weight

    # ------------------------------------------------------------------
    def get_vector_weight(self, vector_id: str) -> float:
        """Return ranking weight for *vector_id* (0.0 if unknown)."""

        conn = self._conn_for(reason="get_vector_weight", commit_required=False)
        cur = conn.execute("SELECT weight FROM vector_weights WHERE vector_id=?", (vector_id,))
        row = cur.fetchone()
        return float(row[0]) if row and row[0] is not None else 0.0

    # ------------------------------------------------------------------
    def set_vector_weight(self, vector_id: str, weight: float) -> None:
        """Persist absolute weight value for *vector_id*."""

        weight = max(0.0, min(1.0, float(weight)))
        conn = self._conn_for(reason="set_vector_weight")
        conn.execute(
            "REPLACE INTO vector_weights(vector_id, weight) VALUES(?, ?)",
            (vector_id, weight),
        )
        conn.commit()

    # ------------------------------------------------------------------
    def recalc_ranking_weights(self) -> Dict[str, float]:
        """Recalculate ranking weights from cumulative ROI and safety data."""

        conn = self._conn_for(reason="recalc_ranking_weights")
        cur = conn.execute(
            """
            SELECT db,
                   COALESCE(SUM(contribution),0) AS roi,
                   COALESCE(AVG(win),0) AS win_rate,
                   COALESCE(AVG(regret),0) AS regret_rate
              FROM vector_metrics
             WHERE event_type='retrieval'
             GROUP BY db
            """
        )
        weights: Dict[str, float] = {}
        for db, roi, win_rate, regret_rate in cur.fetchall():
            roi = float(roi or 0.0)
            win = float(win_rate or 0.0)
            regret = float(regret_rate or 0.0)
            score = roi * max(win, 0.01) * (1.0 - regret)
            if score < 0:
                score = 0.0
            weights[str(db)] = score
        total = sum(weights.values())
        if total > 0:
            weights = {db: w / total for db, w in weights.items()}
        self.set_db_weights(weights)
        return weights

    # ------------------------------------------------------------------
    def save_session(
        self,
        session_id: str,
        vectors: List[Tuple[str, str, float]],
        metadata: Dict[str, Dict[str, Any]],
    ) -> None:
        """Persist session retrieval data for later reconciliation."""

        conn = self._conn_for(reason="save_session")
        conn.execute(
            "REPLACE INTO pending_sessions(session_id, vectors, metadata) VALUES(?,?,?)",
            (session_id, json.dumps(vectors), json.dumps(metadata)),
        )
        conn.commit()

    # ------------------------------------------------------------------
    def load_sessions(
        self,
    ) -> Dict[str, Tuple[List[Tuple[str, str, float]], Dict[str, Dict[str, Any]]]]:
        """Return mapping of session_id to stored vectors and metadata."""

        conn = self._conn_for(reason="load_sessions", commit_required=False)
        cur = conn.execute(
            "SELECT session_id, vectors, metadata FROM pending_sessions"
        )
        sessions: Dict[str, Tuple[List[Tuple[str, str, float]], Dict[str, Dict[str, Any]]]] = {}
        for sid, vec_json, meta_json in cur.fetchall():
            try:
                raw_vecs = json.loads(vec_json or "[]")
                vecs = [
                    (str(o), str(v), float(s))
                    for o, v, s in raw_vecs
                ]
            except Exception:
                vecs = []
            try:
                meta = json.loads(meta_json or "{}")
            except Exception:
                meta = {}
            sessions[str(sid)] = (vecs, meta)
        return sessions

    # ------------------------------------------------------------------
    def delete_session(self, session_id: str) -> None:
        """Remove persisted session data once outcome recorded."""

        conn = self._conn_for(reason="delete_session")
        conn.execute(
            "DELETE FROM pending_sessions WHERE session_id=?",
            (session_id,),
        )
        conn.commit()

    # ------------------------------------------------------------------
    def add(self, rec: VectorMetric) -> None:
        if self._boot_stub_active and self._activate_on_first_write:
            self.activate_persistence(reason="first_write")
            self._activate_on_first_write = False
        if self._boot_stub_active:
            self._buffer_stub_metric(rec)
        if self._should_skip_logging():
            return
        self._ensure_prometheus_ready()
        conn = self._conn_for(reason="add_metric")
        conn.execute(
            """
            INSERT INTO vector_metrics(
                event_type, db, tokens, wall_time_ms, store_time_ms, hit,
                rank, contribution, prompt_tokens, patch_id, session_id,
                vector_id, similarity, context_score, age, win, regret, ts
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                rec.event_type,
                rec.db,
                int(rec.tokens),
                float(rec.wall_time_ms),
                float(rec.store_time_ms),
                None if rec.hit is None else int(rec.hit),
                rec.rank,
                rec.contribution,
                rec.prompt_tokens,
                rec.patch_id,
                rec.session_id,
                rec.vector_id,
                rec.similarity,
                rec.context_score,
                rec.age,
                None if rec.win is None else int(rec.win),
                None if rec.regret is None else int(rec.regret),
                rec.ts,
            ),
        )
        conn.commit()
        if rec.event_type == "embedding":
            try:  # best-effort metrics
                _EMBEDDING_TOKENS_TOTAL.inc(rec.tokens)
            except Exception:
                pass
        elif rec.event_type == "retrieval":
            self._update_retrieval_hit_rate()

    # ------------------------------------------------------------------
    def log_embedding(
        self,
        db: str,
        tokens: int,
        wall_time_ms: float,
        *,
        store_time_ms: float = 0.0,
        prompt_tokens: int | None = None,
        patch_id: str = "",
        vector_id: str = "",
    ) -> None:
        rec = VectorMetric(
            event_type="embedding",
            db=db,
            tokens=tokens,
            wall_time_ms=wall_time_ms,
            store_time_ms=store_time_ms,
            prompt_tokens=prompt_tokens,
            patch_id=patch_id,
            vector_id=vector_id,
        )
        self.add(rec)

    # ------------------------------------------------------------------
    def log_retrieval(
        self,
        db: str,
        tokens: int,
        wall_time_ms: float,
        *,
        hit: bool,
        rank: int,
        contribution: float = 0.0,
        prompt_tokens: int = 0,
        patch_id: str = "",
        store_time_ms: float = 0.0,
        session_id: str = "",
        vector_id: str = "",
        similarity: float = 0.0,
        context_score: float = 0.0,
        age: float = 0.0,
    ) -> None:
        rec = VectorMetric(
            event_type="retrieval",
            db=db,
            tokens=tokens,
            wall_time_ms=wall_time_ms,
            store_time_ms=store_time_ms,
            hit=hit,
            rank=rank,
            contribution=contribution,
            prompt_tokens=prompt_tokens,
            patch_id=patch_id,
            session_id=session_id,
            vector_id=vector_id,
            similarity=similarity,
            context_score=context_score,
            age=age,
        )
        self.add(rec)

    # ------------------------------------------------------------------
    def log_retrieval_feedback(
        self,
        db: str,
        *,
        win: bool = False,
        regret: bool = False,
        roi: float = 0.0,
    ) -> None:
        """Persist aggregate feedback for *db* without session context."""

        rec = VectorMetric(
            event_type="retrieval",
            db=db,
            tokens=0,
            wall_time_ms=0.0,
            contribution=roi,
            win=win,
            regret=regret,
        )
        self.add(rec)

    # ------------------------------------------------------------------
    def log_ranker_update(
        self, db: str, *, delta: float, weight: float | None = None
    ) -> None:
        """Record a ranking weight adjustment for ``db``.

        The ``delta`` reflects the change applied to the weight while
        ``weight`` captures the resulting value when available.  Entries are
        stored in :class:`VectorMetric` with ``event_type`` set to ``"ranker"``
        so historical adjustments can be analysed alongside other vector
        metrics.
        """

        rec = VectorMetric(
            event_type="ranker",
            db=db,
            tokens=0,
            wall_time_ms=0.0,
            contribution=delta,
            similarity=weight,
            context_score=weight,
        )
        self.add(rec)

    # ------------------------------------------------------------------
    def embedding_tokens_total(self, db: str | None = None) -> int:
        conn = self._conn_for(reason="embedding_tokens_total", commit_required=False)
        cur = conn.execute(
            "SELECT COALESCE(SUM(tokens),0) FROM vector_metrics WHERE event_type='embedding'" +
            (" AND db=?" if db else ""),
            (db,) if db else (),
        )
        res = cur.fetchone()
        return int(res[0] if res and res[0] is not None else 0)

    # ------------------------------------------------------------------
    def retrieval_hit_rate(self, db: str | None = None) -> float:
        conn = self._conn_for(reason="retrieval_hit_rate", commit_required=False)
        cur = conn.execute(
            "SELECT AVG(hit) FROM vector_metrics WHERE event_type='retrieval'" +
            (" AND db=?" if db else ""),
            (db,) if db else (),
        )
        res = cur.fetchone()
        return float(res[0]) if res and res[0] is not None else 0.0

    # ------------------------------------------------------------------
    def retriever_win_rate(self, db: str | None = None) -> float:
        conn = self._conn_for(reason="retriever_win_rate", commit_required=False)
        cur = conn.execute(
            "SELECT AVG(win) FROM vector_metrics "
            "WHERE event_type='retrieval' AND win IS NOT NULL"
            + (" AND db=?" if db else ""),
            (db,) if db else (),
        )
        res = cur.fetchone()
        return float(res[0]) if res and res[0] is not None else 0.0

    # ------------------------------------------------------------------
    def retriever_regret_rate(self, db: str | None = None) -> float:
        conn = self._conn_for(reason="retriever_regret_rate", commit_required=False)
        cur = conn.execute(
            "SELECT AVG(regret) FROM vector_metrics "
            "WHERE event_type='retrieval' AND regret IS NOT NULL"
            + (" AND db=?" if db else ""),
            (db,) if db else (),
        )
        res = cur.fetchone()
        return float(res[0]) if res and res[0] is not None else 0.0

    # ------------------------------------------------------------------
    def retriever_win_rate_by_db(self) -> dict[str, float]:
        self._ensure_prometheus_ready()
        conn = self._conn_for(reason="retriever_win_rate_by_db", commit_required=False)
        cur = conn.execute(
            """
            SELECT db, AVG(win)
              FROM vector_metrics
             WHERE event_type='retrieval' AND win IS NOT NULL
             GROUP BY db
            """
        )
        rows = cur.fetchall()
        rates = {str(db): float(rate) if rate is not None else 0.0 for db, rate in rows}
        for name, rate in rates.items():
            try:
                _RETRIEVER_WIN_RATE.labels(db=name).set(rate)
            except Exception:
                pass
        return rates

    # ------------------------------------------------------------------
    def retriever_regret_rate_by_db(self) -> dict[str, float]:
        self._ensure_prometheus_ready()
        conn = self._conn_for(
            reason="retriever_regret_rate_by_db", commit_required=False
        )
        cur = conn.execute(
            """
            SELECT db, AVG(regret)
              FROM vector_metrics
             WHERE event_type='retrieval' AND regret IS NOT NULL
             GROUP BY db
            """
        )
        rows = cur.fetchall()
        rates = {str(db): float(rate) if rate is not None else 0.0 for db, rate in rows}
        for name, rate in rates.items():
            try:
                _RETRIEVER_REGRET_RATE.labels(db=name).set(rate)
            except Exception:
                pass
        return rates

    # ------------------------------------------------------------------
    def update_outcome(
        self,
        session_id: str,
        vectors: list[tuple[str, str]],
        *,
        contribution: float,
        patch_id: str = "",
        win: bool = False,
        regret: bool = False,
    ) -> None:
        if self._should_skip_logging():
            return
        conn = self._conn_for(reason="update_outcome")
        for _, vec_id in vectors:
            conn.execute(
                """
                UPDATE vector_metrics
                   SET contribution=?, win=?, regret=?, patch_id=?
                 WHERE session_id=? AND vector_id=?
                """,
                (
                    contribution,
                    int(win),
                    int(regret),
                    patch_id,
                    session_id,
                    vec_id,
                ),
            )
        conn.commit()

    def record_patch_ancestry(
        self, patch_id: str, vectors: list[tuple]
    ) -> None:
        if self._should_skip_logging():
            return
        conn = self._conn_for(reason="record_patch_ancestry")
        for rank, vec in enumerate(vectors):
            vec_id, contrib, lic, alerts, sev, risk = (
                list(vec) + [None, None, None, None, None]
            )[:6]
            conn.execute(
                "INSERT INTO patch_ancestry(patch_id, vector_id, rank, contribution, "
                "license, semantic_alerts, alignment_severity, risk_score) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (
                    patch_id,
                    vec_id,
                    rank,
                    contrib,
                    lic,
                    json.dumps(alerts) if alerts is not None else None,
                    sev,
                    risk,
                ),
            )
        conn.commit()

    def record_patch_summary(
        self,
        patch_id: str,
        *,
        errors: Sequence[Mapping[str, Any]] | None = None,
        tests_passed: bool | None = None,
        lines_changed: int | None = None,
        context_tokens: int | None = None,
        patch_difficulty: int | None = None,
        start_time: float | None = None,
        time_to_completion: float | None = None,
        error_trace_count: int | None = None,
        roi_tag: str | None = None,
        effort_estimate: float | None = None,
        enhancement_score: float | None = None,
    ) -> None:
        if self._should_skip_logging():
            return
        try:
            conn = self._conn_for(reason="record_patch_summary")
            conn.execute(
                "REPLACE INTO patch_metrics(patch_id, errors, tests_passed, "
                "lines_changed, context_tokens, patch_difficulty, start_time, "
                "time_to_completion, error_trace_count, roi_tag, "
                "effort_estimate, enhancement_score) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    patch_id,
                    json.dumps(list(errors or [])),
                    None if tests_passed is None else int(bool(tests_passed)),
                    lines_changed,
                    context_tokens,
                    patch_difficulty,
                    start_time,
                    time_to_completion,
                    error_trace_count,
                    roi_tag,
                    effort_estimate,
                    enhancement_score,
                ),
            )
            conn.commit()
        except Exception:
            logging.getLogger(__name__).exception("failed to record patch summary")

    # ------------------------------------------------------------------
    def _update_retrieval_hit_rate(self) -> None:
        self._ensure_prometheus_ready()
        try:  # best-effort metrics
            _RETRIEVAL_HIT_RATE.set(self.retrieval_hit_rate())
        except Exception:
            pass


__all__ = [
    "VectorMetric",
    "VectorMetricsDB",
    "resolve_vector_bootstrap_flags",
    "get_bootstrap_vector_metrics_db",
    "get_shared_vector_metrics_db",
    "activate_shared_vector_metrics_db",
    "ensure_vector_db_weights",
]
