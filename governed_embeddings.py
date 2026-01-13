from __future__ import annotations

import contextlib
import errno
import inspect
import logging
import os
import shutil
import sys
import tarfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, List, Mapping, MutableMapping, Optional, TYPE_CHECKING, cast, Set

try:  # pragma: no cover - lock utilities are optional in some environments
    from lock_utils import (
        LOCK_TIMEOUT,
        SandboxLock,
        Timeout as LockTimeout,
        is_lock_stale,
    )
except Exception:  # pragma: no cover - degrade gracefully when locks unavailable
    SandboxLock = None  # type: ignore[assignment]

    class _FallbackTimeout(Exception):
        """Placeholder used when :mod:`filelock` is unavailable."""

        pass

    LockTimeout = _FallbackTimeout  # type: ignore[assignment]
    LOCK_TIMEOUT = float(os.getenv("LOCK_TIMEOUT", "60"))
    is_lock_stale = None  # type: ignore[assignment]

if SandboxLock is None:  # pragma: no cover - fallback when lock_utils unavailable
    from fcntl_compat import LOCK_EX, LOCK_NB, LOCK_UN, flock

    class _SimpleLockGuard:
        """Context manager implementing ``with lock.acquire():`` semantics."""

        def __init__(self, lock: "_SimpleSandboxLock", timeout: float | None) -> None:
            if timeout is None or timeout < 0:
                timeout = LOCK_TIMEOUT
            self._timeout = timeout
            self._lock = lock

        def __enter__(self) -> "_SimpleSandboxLock":
            self._lock._acquire(self._timeout)
            return self._lock

        def __exit__(self, exc_type, exc, tb) -> None:
            self._lock.release()

    class _SimpleSandboxLock:
        """Minimal cross-process file lock used when :mod:`lock_utils` is absent."""

        def __init__(self, lock_file: str) -> None:
            self.lock_file = lock_file
            self.timeout = LOCK_TIMEOUT
            self._fd: int | None = None
            self._guard: _SimpleLockGuard | None = None

        # ------------------------------------------------------------------ helpers
        def _acquire(self, timeout: float) -> None:
            deadline = None if timeout < 0 else time.monotonic() + timeout
            os.makedirs(os.path.dirname(self.lock_file), exist_ok=True)
            poll_interval = 0.1
            while True:
                fd = os.open(self.lock_file, os.O_RDWR | os.O_CREAT, 0o644)
                try:
                    flock(fd, LOCK_EX | LOCK_NB)
                except OSError as exc:
                    os.close(fd)
                    if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK):
                        raise
                    if deadline is not None and time.monotonic() >= deadline:
                        raise LockTimeout(self.lock_file)
                    time.sleep(poll_interval)
                    continue

                self._fd = fd
                try:
                    os.ftruncate(fd, 0)
                    os.write(fd, f"{os.getpid()},{time.time()}".encode("ascii", "ignore"))
                except Exception:  # pragma: no cover - metadata best effort
                    pass
                return

        # ---------------------------------------------------------------- context API
        def acquire(self, timeout: float | None = None) -> _SimpleLockGuard:
            return _SimpleLockGuard(self, timeout)

        def release(self) -> None:
            if self._fd is None:
                return
            fd = self._fd
            self._fd = None
            try:
                flock(fd, LOCK_UN)
            except Exception:  # pragma: no cover - best effort
                pass
            finally:
                try:
                    os.close(fd)
                except Exception:  # pragma: no cover - defensive
                    pass
            try:
                os.remove(self.lock_file)
            except FileNotFoundError:
                pass
            except Exception as exc:  # pragma: no cover - diagnostics only
                logger.debug("failed to remove fallback lock file %s: %s", self.lock_file, exc)

        def __enter__(self) -> "_SimpleSandboxLock":
            self._guard = self.acquire()
            self._guard.__enter__()
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            if self._guard is not None:
                self._guard.__exit__(exc_type, exc, tb)
                self._guard = None
            else:  # pragma: no cover - defensive
                self.release()

    SandboxLock = _SimpleSandboxLock  # type: ignore[assignment]
    SandboxLockType = _SimpleSandboxLock  # type: ignore[assignment]
    logging.getLogger(__name__).debug(
        "lock_utils unavailable; using simplified sandbox lock"
    )


if TYPE_CHECKING:  # pragma: no cover - typing helper
    from lock_utils import SandboxLock as SandboxLockType
else:  # pragma: no cover - at runtime we either have the real class or ``Any``
    SandboxLockType = Any

model: "SentenceTransformer | None"

DEFAULT_SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_CHAR_TRUNCATION_THRESHOLD = 8000
EMBEDDING_CHARS_PER_TOKEN = 4


def canonical_model_id(model_name: str | None) -> str:
    """Return a canonical ``SentenceTransformer`` repo identifier."""

    name = (model_name or "").strip()
    if not name:
        return DEFAULT_SENTENCE_TRANSFORMER_MODEL
    if "/" not in name:
        return f"sentence-transformers/{name}"
    if name.lower().startswith("sentence-transformers/"):
        return name
    return name


_MODEL_ID = canonical_model_id(DEFAULT_SENTENCE_TRANSFORMER_MODEL)

try:  # pragma: no cover - optional heavy dependency
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - simplify in environments without the package
    SentenceTransformer = None  # type: ignore
    model = None
else:
    # ``SentenceTransformer`` can trigger network downloads when instantiated.
    # Avoid constructing it at import time so tests or simple imports do not
    # block waiting for external resources.  The instance will be created lazily
    # by ``_load_embedder`` when embeddings are actually required.
    model = None


def _resolve_sentence_transformer_device() -> str:
    """Return the preferred device for ``SentenceTransformer`` instances."""

    env_device = os.getenv("SENTENCE_TRANSFORMER_DEVICE", "").strip()
    if env_device:
        return env_device

    # Default to CPU to avoid invoking GPU-specific code paths which can be
    # fragile across environments.  Callers can still opt in to a different
    # device by setting ``SENTENCE_TRANSFORMER_DEVICE`` explicitly.
    return "cpu"

try:  # pragma: no cover - lightweight fallback dependencies
    import torch
    from transformers import AutoModel, AutoTokenizer
except Exception:  # pragma: no cover - defer heavy imports until needed
    torch = None  # type: ignore[assignment]
    AutoModel = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]

from security.secret_redactor import redact
from compliance.license_fingerprint import (
    check as license_check,
    fingerprint as license_fingerprint,
)
from analysis.semantic_diff_filter import find_semantic_risks

logger = logging.getLogger(__name__)

_EMBEDDER: SentenceTransformer | None = None
_EMBEDDER_LOCK: SandboxLockType | None = None
_EMBEDDER_THREAD_LOCK = threading.RLock()
_MODEL_NAME = _MODEL_ID
SENTENCE_TRANSFORMER_DEVICE = _resolve_sentence_transformer_device()
_EMBEDDER_INIT_TIMEOUT = float(os.getenv("EMBEDDER_INIT_TIMEOUT", "180"))
_MAX_EMBEDDER_WAIT = float(os.getenv("EMBEDDER_INIT_MAX_WAIT", "180"))
_SOFT_EMBEDDER_WAIT = float(os.getenv("EMBEDDER_INIT_SOFT_WAIT", "30"))
_EMBEDDER_INIT_EVENT = threading.Event()
_EMBEDDER_INIT_THREAD: threading.Thread | None = None
_EMBEDDER_STOP_EVENT: threading.Event | None = None
_EMBEDDER_DISABLED = False
_EMBEDDER_TIMEOUT_LOGGED = False
_EMBEDDER_WAIT_CAPPED = False
_EMBEDDER_SOFT_WAIT_LOGGED = False
_EMBEDDER_TIMEOUT_REACHED = False
_EMBEDDER_BOOTSTRAP_DEFERRED = False
_EMBEDDER_BOOTSTRAP_PLACEHOLDER: Any | None = None
_EMBEDDER_BOOTSTRAP_DEFERRALS: Set[str] = set()
_EMBEDDER_BOOTSTRAP_DEFERRALS_LOCK = threading.Lock()
_FALLBACK_ANNOUNCED = False
_HF_LOCK_CLEANUP_TIMEOUT = float(os.getenv("HF_LOCK_CLEANUP_TIMEOUT", "5"))
_BUNDLED_EMBEDDER: Any | None = None
_BUNDLED_EMBEDDER_LOCK = threading.Lock()
_EMBEDDER_REQUESTER_TIMEOUTS: Set[str] = set()
_EMBEDDER_REQUESTER_LOGGED: Set[str] = set()
_EMBEDDER_REQUESTER_LOCK = threading.Lock()
_STUB_EMBEDDER_DIMENSION = 384
_STUB_FALLBACK_USED = False
_HF_TIMEOUT_SETTINGS: dict[str, str] = {}
_HF_TIMEOUT_CONFIGURED = False
_FORCED_LOCK_CLEANUP_GUARD = threading.Lock()
_STACKTRACE_LOCK = threading.Lock()
_EMBEDDER_STACK_REPORTED: Set[str] = set()
_BOOTSTRAP_EMBEDDER_WAIT_DEFAULT = 5.0


def _determine_timeout_cap() -> float:
    """Return the maximum allowed embedder wait based on configuration."""

    default_cap = 300.0
    raw = os.getenv("EMBEDDER_INIT_TIMEOUT_CAP", "").strip()
    if not raw:
        return default_cap
    try:
        cap = float(raw)
    except Exception:
        logger.warning(
            "invalid EMBEDDER_INIT_TIMEOUT_CAP=%r; defaulting to %.0fs",
            raw,
            default_cap,
        )
        return default_cap
    return cap


def _cap_timeout(value: float, cap: float, env_name: str) -> float:
    """Clamp ``value`` to ``cap`` when both are non-negative."""

    if value < 0 or cap < 0:
        return value
    if value <= cap:
        return value
    logger.warning(
        "capping %s to %.0fs (requested %.0fs)",
        env_name,
        cap,
        value,
    )
    return cap


def _apply_timeout_caps() -> None:
    """Normalise embedder wait settings to avoid multi-hour stalls."""

    global _EMBEDDER_INIT_TIMEOUT, _MAX_EMBEDDER_WAIT, _SOFT_EMBEDDER_WAIT, _EMBEDDER_TIMEOUT_CAP

    cap = _determine_timeout_cap()
    _EMBEDDER_TIMEOUT_CAP = cap
    _EMBEDDER_INIT_TIMEOUT = _cap_timeout(
        _EMBEDDER_INIT_TIMEOUT, cap, "EMBEDDER_INIT_TIMEOUT"
    )
    _MAX_EMBEDDER_WAIT = _cap_timeout(
        _MAX_EMBEDDER_WAIT, cap, "EMBEDDER_INIT_MAX_WAIT"
    )
    if (
        _SOFT_EMBEDDER_WAIT >= 0
        and _MAX_EMBEDDER_WAIT >= 0
        and _SOFT_EMBEDDER_WAIT > _MAX_EMBEDDER_WAIT
    ):
        logger.warning(
            "capping EMBEDDER_INIT_SOFT_WAIT to %.0fs (requested %.0fs)",
            _MAX_EMBEDDER_WAIT,
            _SOFT_EMBEDDER_WAIT,
        )
        _SOFT_EMBEDDER_WAIT = _MAX_EMBEDDER_WAIT


_EMBEDDER_TIMEOUT_CAP = 0.0
_apply_timeout_caps()


def _bootstrap_context_timeout_cap() -> float:
    """Return the hard cap to apply when running under bootstrap-like flows."""

    configured = os.getenv("EMBEDDER_BOOTSTRAP_WAIT", "").strip()
    fallback_cap = _BOOTSTRAP_EMBEDDER_WAIT_DEFAULT
    try:
        if configured:
            fallback_cap = max(0.0, float(configured))
    except Exception:
        logger.debug(
            "ignoring invalid EMBEDDER_BOOTSTRAP_WAIT=%r", configured
        )

    if _SOFT_EMBEDDER_WAIT >= 0:
        fallback_cap = min(fallback_cap, _SOFT_EMBEDDER_WAIT)
    return fallback_cap


def apply_bootstrap_timeout_caps(budget: float | None = None) -> float:
    """Clamp embedder waits to a small budget for bootstrap/warmup paths."""

    global _EMBEDDER_INIT_TIMEOUT, _MAX_EMBEDDER_WAIT

    cap = _bootstrap_context_timeout_cap()
    if budget is not None and budget >= 0:
        cap = min(cap, budget)
    if _EMBEDDER_INIT_TIMEOUT >= 0 and _EMBEDDER_INIT_TIMEOUT > cap:
        logger.info(
            "tightening embedder init timeout to %.1fs for bootstrap/warmup (requested %.1fs)",
            cap,
            _EMBEDDER_INIT_TIMEOUT,
        )
        _EMBEDDER_INIT_TIMEOUT = cap
    if _MAX_EMBEDDER_WAIT >= 0 and _MAX_EMBEDDER_WAIT > cap:
        logger.info(
            "tightening embedder max wait to %.1fs for bootstrap/warmup (requested %.1fs)",
            cap,
            _MAX_EMBEDDER_WAIT,
        )
        _MAX_EMBEDDER_WAIT = cap
    return cap


def _stacktrace_enabled() -> bool:
    flag = os.getenv("EMBEDDER_STACKTRACE", "").strip().lower()
    return bool(flag) and flag not in {"0", "false", "no", "off"}


def _dump_embedder_thread(reason: str, waited: float | None = None) -> None:
    """Log the current stack of the embedder thread when diagnostics are enabled."""

    if not _stacktrace_enabled():
        return

    thread = _EMBEDDER_INIT_THREAD
    if thread is None or not thread.is_alive():
        return

    ident = thread.ident
    if ident is None:
        return

    key = f"{ident}:{reason}"
    with _STACKTRACE_LOCK:
        if key in _EMBEDDER_STACK_REPORTED:
            return
        _EMBEDDER_STACK_REPORTED.add(key)

    try:
        frames = sys._current_frames()
    except Exception:  # pragma: no cover - diagnostic best effort
        return

    frame = frames.get(ident)
    if frame is None:
        return

    stack = "".join(traceback.format_stack(frame))
    if waited is not None:
        logger.warning(
            "embedder initialisation thread stack trace (%s, waited %.1fs):\n%s",
            reason,
            waited,
            stack,
        )
    else:
        logger.warning(
            "embedder initialisation thread stack trace (%s):\n%s",
            reason,
            stack,
        )
    _trace(
        "thread.stack_dump",
        reason=reason,
        waited=round(waited, 3) if waited is not None else None,
    )


def _trace(event: str, **extra: Any) -> None:
    """Emit verbose diagnostics when ``EMBEDDER_TRACE`` is enabled."""

    flag = os.getenv("EMBEDDER_TRACE", "").strip().lower()
    if not flag or flag in {"0", "false", "no", "off"}:
        return
    payload = {"event": event, **extra}
    logger.log(logging.INFO, "embedder: %s", event, extra=payload)


def _bundled_model_archive() -> Path | None:
    base = Path(__file__).resolve().parent
    archive = base / "vector_service" / "minilm" / "tiny-distilroberta-base.tar.xz"
    return archive if archive.exists() else None


def _prepare_bundled_model_dir() -> Path | None:
    archive = _bundled_model_archive()
    if archive is None:
        logger.debug("bundled embedder archive missing")
        return None

    cache_dir = _cache_base()
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "huggingface"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # pragma: no cover - best effort
        logger.debug("failed to prepare cache directory for bundled embedder: %s", exc)
        return None

    target_dir = cache_dir / "menace-bundled" / "tiny-distilroberta-base"
    sentinel = target_dir / "config.json"
    if not sentinel.exists():
        tmp_dir = target_dir.with_suffix(".tmp")
        try:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            tmp_dir.mkdir(parents=True, exist_ok=True)
            _extract_bundled_archive(archive, tmp_dir)
            if target_dir.exists():
                shutil.rmtree(target_dir, ignore_errors=True)
            tmp_dir.rename(target_dir)
        except Exception as exc:
            logger.warning("failed to extract bundled embedder archive: %s", exc)
            _signal_vector_readiness_failure(
                "bundled_embedder_extract_failed", error=str(exc)
            )
            with contextlib.suppress(Exception):
                shutil.rmtree(tmp_dir, ignore_errors=True)
            return None
    return target_dir


def _bundled_extract_timeout() -> float:
    raw_timeout = os.getenv("MENACE_BUNDLED_EMBEDDER_EXTRACT_TIMEOUT", "").strip()
    if not raw_timeout:
        return 20.0
    try:
        parsed = float(raw_timeout)
    except Exception:
        logger.warning(
            "invalid MENACE_BUNDLED_EMBEDDER_EXTRACT_TIMEOUT=%r; defaulting to 20s",
            raw_timeout,
        )
        return 20.0
    if parsed <= 0:
        logger.warning(
            "MENACE_BUNDLED_EMBEDDER_EXTRACT_TIMEOUT must be positive; defaulting to 20s"
        )
        return 20.0
    return parsed


def _signal_vector_readiness_failure(reason: str, *, error: str | None = None) -> None:
    _update_vector_readiness_status("failed", reason=reason, error=error)


def _update_vector_readiness_status(
    status: str, *, remaining: float | None = None, reason: str | None = None, error: str | None = None
) -> None:
    try:  # pragma: no cover - defensive best effort
        from bootstrap_timeout_policy import emit_bootstrap_heartbeat, read_bootstrap_heartbeat
    except Exception:
        return

    try:  # pragma: no cover - lightweight bookkeeping
        from bootstrap_readiness import CORE_COMPONENTS
    except Exception:
        CORE_COMPONENTS = ()

    heartbeat = read_bootstrap_heartbeat() or {}
    readiness = heartbeat.get("readiness") if isinstance(heartbeat, Mapping) else {}
    components: MutableMapping[str, str] = {}
    component_readiness: MutableMapping[str, Mapping[str, object]] = {}

    if isinstance(readiness, Mapping):
        raw_components = readiness.get("components")
        if isinstance(raw_components, Mapping):
            components.update({str(key): str(value) for key, value in raw_components.items()})

        raw_component_readiness = readiness.get("component_readiness")
        if isinstance(raw_component_readiness, Mapping):
            for key, value in raw_component_readiness.items():
                state = value if isinstance(value, Mapping) else {"status": value}
                component_readiness[str(key)] = dict(state)

    now = time.time()
    components["vector_seeding"] = status
    detail: MutableMapping[str, object] = {"status": status, "ts": now}
    if reason:
        detail["reason"] = reason
    if error:
        detail["error"] = error
    if remaining is not None:
        detail["remaining"] = max(0.0, remaining)
    component_readiness["vector_seeding"] = dict(detail)

    all_ready = all(components.get(name) == "ready" for name in CORE_COMPONENTS)
    readiness_payload: dict[str, object] = {
        "components": dict(components),
        "component_readiness": dict(component_readiness),
        "ready": all_ready,
        "online": bool(all_ready or (isinstance(readiness, Mapping) and readiness.get("online"))),
    }

    enriched = dict(heartbeat)
    enriched["readiness"] = readiness_payload

    emit_bootstrap_heartbeat(enriched)


def _extract_bundled_archive(archive: Path, dest: Path) -> None:
    timeout = _bundled_extract_timeout()
    stop_event = threading.Event()
    done_event = threading.Event()
    result: dict[str, Exception | None] = {"error": None}
    deadline = time.perf_counter() + timeout if timeout > 0 else None

    def _run() -> None:
        try:
            with tarfile.open(archive) as tar:
                members = tar.getmembers()
                next_heartbeat = time.perf_counter() + 2.5
                for idx, member in enumerate(members, 1):
                    if stop_event.is_set():
                        raise TimeoutError("bundled embedder extraction cancelled")
                    tar.extract(member, dest)
                    now = time.perf_counter()
                    if now >= next_heartbeat:
                        remaining = None if deadline is None else max(0.0, deadline - now)
                        _update_vector_readiness_status(
                            "extracting",
                            remaining=remaining,
                            reason="bundled_embedder_extract",
                            error=None,
                        )
                        next_heartbeat = now + 2.5
        except Exception as exc:  # pragma: no cover - diagnostic path
            result["error"] = exc
        finally:
            done_event.set()

    _update_vector_readiness_status(
        "extracting",
        remaining=timeout if timeout > 0 else None,
        reason="bundled_embedder_extract",
        error=None,
    )

    thread = threading.Thread(
        target=_run,
        name="bundled-embedder-extract",
        daemon=True,
    )
    thread.start()

    while not done_event.wait(0.5):
        if deadline is not None and time.perf_counter() >= deadline:
            stop_event.set()
            break
        if deadline is not None:
            _update_vector_readiness_status(
                "extracting",
                remaining=max(0.0, deadline - time.perf_counter()),
                reason="bundled_embedder_extract",
                error=None,
            )

    if stop_event.is_set():
        logger.warning(
            "bundled embedder extraction exceeded timeout; cancelling background task",
            extra={"archive": str(archive), "timeout": timeout},
        )
        _signal_vector_readiness_failure("bundled_embedder_extract_timeout")
        raise TimeoutError("bundled embedder extraction timed out")

    done_event.wait(2.0)
    if result["error"]:
        raise result["error"]

    _update_vector_readiness_status(
        "warming",
        remaining=None,
        reason="bundled_embedder_extract_complete",
        error=None,
    )



def _build_stub_embedder() -> Any:
    """Return a minimal stand-in for :class:`SentenceTransformer`."""

    try:  # pragma: no cover - optional dependency
        import numpy as np
    except Exception:  # pragma: no cover - numpy may be unavailable
        np = None  # type: ignore[assignment]

    class _StubSentenceTransformer:
        def __init__(self, dimension: int) -> None:
            self._dimension = dimension

        def encode(self, sentences: Any, **kwargs: Any) -> Any:
            if isinstance(sentences, str):
                sentences = [sentences]
            convert_to_numpy = kwargs.get("convert_to_numpy", np is not None)
            batch = [[0.0] * self._dimension for _ in sentences]
            if convert_to_numpy and np is not None:
                return np.asarray(batch, dtype="float32")
            return batch

        def get_sentence_embedding_dimension(self) -> int:
            return self._dimension

    return _StubSentenceTransformer(_STUB_EMBEDDER_DIMENSION)


def _record_bootstrap_placeholder(reason: str) -> Any:
    """Memoize and return a placeholder embedder for bootstrap callers."""

    global _EMBEDDER_BOOTSTRAP_DEFERRED, _EMBEDDER_BOOTSTRAP_PLACEHOLDER

    if _EMBEDDER_BOOTSTRAP_PLACEHOLDER is None:
        _EMBEDDER_BOOTSTRAP_PLACEHOLDER = _build_stub_embedder()
    setattr(_EMBEDDER_BOOTSTRAP_PLACEHOLDER, "_placeholder_reason", reason)
    if not _EMBEDDER_BOOTSTRAP_DEFERRED:
        logger.info(
            "bootstrap embedder warmup deferred; returning placeholder",
            extra={"model": _MODEL_NAME, "reason": reason},
        )
        _trace("bootstrap.deferred", reason=reason)
    _EMBEDDER_BOOTSTRAP_DEFERRED = True
    return _EMBEDDER_BOOTSTRAP_PLACEHOLDER


def _bootstrap_budget_remaining(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    return deadline - time.perf_counter()


def _bootstrap_background_executor() -> Any | None:
    try:  # pragma: no cover - optional dependency
        from vector_service.lazy_bootstrap import _background_executor  # type: ignore
    except Exception as exc:  # pragma: no cover - best effort diagnostics
        logger.debug("bootstrap background executor unavailable: %s", exc)
        _trace("bootstrap.executor.unavailable", error=str(exc))
        return None

    try:
        return _background_executor()
    except Exception as exc:  # pragma: no cover - diagnostic only
        logger.debug("failed to access bootstrap executor: %s", exc)
        _trace("bootstrap.executor.error", error=str(exc))
        return None


def _schedule_deferred_embedder_load(
    *,
    reason: str,
    requester: str | None,
) -> bool:
    executor = _bootstrap_background_executor()
    if executor is None:
        return False

    background_event = threading.Event()

    def _resume() -> None:
        _ensure_embedder_thread_locked(
            background_event,
            load_context={
                "budget_deadline": None,
                "bootstrap_mode": False,
                "requester": requester or "bootstrap-background",
            },
        )

    try:
        executor.submit(_resume)
    except Exception as exc:  # pragma: no cover - background scheduling best effort
        logger.debug("failed to schedule deferred embedder load: %s", exc)
        _trace("bootstrap.executor.submit_error", error=str(exc), reason=reason)
        return False

    return True


def _defer_bootstrap_load(
    *,
    reason: str,
    requester: str | None = None,
) -> Any:
    scheduled = False
    with _EMBEDDER_BOOTSTRAP_DEFERRALS_LOCK:
        first_seen = reason not in _EMBEDDER_BOOTSTRAP_DEFERRALS
        _EMBEDDER_BOOTSTRAP_DEFERRALS.add(reason)
    if first_seen or not _EMBEDDER_BOOTSTRAP_DEFERRED:
        scheduled = _schedule_deferred_embedder_load(
            reason=reason,
            requester=requester,
        )
    placeholder = _record_bootstrap_placeholder(reason)
    if scheduled:
        logger.info(
            "embedder load deferred (%s); scheduled background bootstrap load",
            reason,
            extra={"model": _MODEL_NAME, "requester": requester},
        )
    _trace(
        "load.deferred", reason=reason, scheduled=scheduled, requester=requester
    )
    return placeholder


def _guard_bootstrap_budget(
    *,
    budget_deadline: float | None,
    stop_event: threading.Event | None,
    bootstrap_mode: bool,
    requester: str | None,
    stage: str,
) -> Any | None:
    remaining = _bootstrap_budget_remaining(budget_deadline)
    if not bootstrap_mode:
        return None

    if _embedder_stop_requested(stop_event):
        return _defer_bootstrap_load(
            reason=f"{stage}_cancelled",
            requester=requester,
        )

    if remaining is not None and remaining <= 0:
        if stop_event is not None:
            stop_event.set()
        return _defer_bootstrap_load(
            reason=f"{stage}_budget_exhausted",
            requester=requester,
        )

    return None


def _load_bundled_embedder() -> Any | None:
    global _BUNDLED_EMBEDDER
    if _BUNDLED_EMBEDDER is not None:
        return _BUNDLED_EMBEDDER

    archive = _bundled_model_archive()
    if archive is None:
        global _STUB_FALLBACK_USED
        if not _STUB_FALLBACK_USED:
            _STUB_FALLBACK_USED = True
            logger.warning(
                "bundled embedder archive missing; using stub sentence transformer"
            )
        _BUNDLED_EMBEDDER = _build_stub_embedder()
        return _BUNDLED_EMBEDDER
    if torch is None or AutoModel is None or AutoTokenizer is None:  # pragma: no cover - optional deps
        logger.warning("bundled embedder unavailable because transformers stack is missing; using stub sentence transformer")
        _BUNDLED_EMBEDDER = _build_stub_embedder()
        return _BUNDLED_EMBEDDER

    model_dir = _prepare_bundled_model_dir()
    if model_dir is None:
        logger.warning(
            "failed to prepare bundled embedder directory; using stub sentence transformer"
        )
        _BUNDLED_EMBEDDER = _build_stub_embedder()
        return _BUNDLED_EMBEDDER

    with _BUNDLED_EMBEDDER_LOCK:
        if _BUNDLED_EMBEDDER is not None:
            return _BUNDLED_EMBEDDER

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = AutoModel.from_pretrained(model_dir)
            model.eval()
        except Exception as exc:  # pragma: no cover - diagnostics only
            logger.warning("failed to load bundled fallback embedder: %s; using stub sentence transformer", exc)
            _BUNDLED_EMBEDDER = _build_stub_embedder()
            return _BUNDLED_EMBEDDER

        class _FallbackSentenceTransformer:
            """Minimal wrapper emulating the ``SentenceTransformer`` API."""

            def __init__(self, tok, mdl) -> None:
                self._tokenizer = tok
                self._model = mdl

            def encode(self, sentences: Any, **kwargs: Any) -> List[List[float]]:
                if isinstance(sentences, str):
                    batch = [sentences]
                else:
                    batch = list(sentences)
                if not batch:
                    return []
                inputs = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                with torch.no_grad():
                    outputs = self._model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                if kwargs.get("normalize_embeddings"):
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                return embeddings.cpu().tolist()

            def get_sentence_embedding_dimension(self) -> int:
                try:
                    return int(self._model.config.hidden_size)
                except Exception:  # pragma: no cover - defensive
                    sample = self.encode(["test"])
                    return len(sample[0]) if sample else 0

        _BUNDLED_EMBEDDER = _FallbackSentenceTransformer(tokenizer, model)
        logger.info(
            "using bundled fallback sentence transformer",
            extra={"archive": str(archive)},
        )
        return _BUNDLED_EMBEDDER


def _hf_timeout_values() -> tuple[float, int]:
    """Return the configured Hugging Face timeout and retry values."""

    global _HF_TIMEOUT_SETTINGS
    if _HF_TIMEOUT_SETTINGS:
        timeout = float(_HF_TIMEOUT_SETTINGS.get("timeout", "30"))
        retries = int(_HF_TIMEOUT_SETTINGS.get("retries", "1"))
        return timeout, retries

    raw_timeout = os.getenv("EMBEDDER_HF_TIMEOUT", "").strip()
    timeout = 30.0
    if raw_timeout:
        try:
            timeout = float(raw_timeout)
        except Exception:
            logger.warning(
                "invalid EMBEDDER_HF_TIMEOUT=%r; defaulting to 30s", raw_timeout
            )
            timeout = 30.0
    if timeout <= 0:
        logger.warning(
            "EMBEDDER_HF_TIMEOUT must be positive; defaulting to 30s"
        )
        timeout = 30.0

    raw_retries = os.getenv("EMBEDDER_HF_RETRIES", "").strip()
    retries = 1
    if raw_retries:
        try:
            retries = int(float(raw_retries))
        except Exception:
            logger.warning(
                "invalid EMBEDDER_HF_RETRIES=%r; defaulting to 1", raw_retries
            )
            retries = 1
    if retries <= 0:
        logger.warning("EMBEDDER_HF_RETRIES must be positive; defaulting to 1")
        retries = 1

    _HF_TIMEOUT_SETTINGS = {"timeout": f"{timeout:g}", "retries": str(retries)}
    return timeout, retries


def _ensure_hf_timeouts() -> None:
    """Apply conservative Hugging Face hub timeouts to avoid long stalls."""

    global _HF_TIMEOUT_CONFIGURED
    if _HF_TIMEOUT_CONFIGURED:
        return

    if os.getenv("EMBEDDER_DISABLE_HF_TIMEOUTS", "").lower() in {
        "1",
        "true",
        "yes",
    }:
        _HF_TIMEOUT_CONFIGURED = True
        return

    timeout, retries = _hf_timeout_values()
    timeout_str = f"{timeout:g}"
    applied: dict[str, str] = {}

    for env_name in ("HF_HUB_TIMEOUT", "HF_HUB_READ_TIMEOUT", "HF_HUB_CONNECTION_TIMEOUT"):
        if not os.environ.get(env_name):
            os.environ[env_name] = timeout_str
            applied[env_name] = timeout_str

    if not os.environ.get("HF_HUB_DOWNLOAD_RETRIES"):
        os.environ["HF_HUB_DOWNLOAD_RETRIES"] = str(retries)
        applied["HF_HUB_DOWNLOAD_RETRIES"] = str(retries)

    if applied:
        logger.debug(
            "configured huggingface hub timeouts", extra={"settings": applied}
        )

    _HF_TIMEOUT_CONFIGURED = True


def _activate_bundled_fallback(reason: str) -> bool:
    """Promote the bundled embedder to the shared embedder slot.

    The helper is invoked when the hub initialisation thread exceeds its
    timeout.  It loads the lightweight bundled model (if available) and marks
    the shared embedder as ready so dependant services can proceed instead of
    blocking indefinitely while the background thread continues its best-effort
    hub initialisation.
    """

    global _EMBEDDER, _FALLBACK_ANNOUNCED

    fallback = _load_bundled_embedder()
    if fallback is None:
        return False

    with _EMBEDDER_THREAD_LOCK:
        if _EMBEDDER is None:
            _EMBEDDER = cast("SentenceTransformer", fallback)
            _EMBEDDER_INIT_EVENT.set()
            if not _FALLBACK_ANNOUNCED:
                _FALLBACK_ANNOUNCED = True
                logger.warning(
                    "using bundled fallback sentence transformer while hub initialisation continues",
                    extra={"model": _MODEL_NAME, "reason": reason},
                )
            return True

    if _EMBEDDER is not None:
        return True

    if not _FALLBACK_ANNOUNCED:
        _FALLBACK_ANNOUNCED = True
        logger.warning(
            "bundled fallback sentence transformer already active",
            extra={"model": _MODEL_NAME, "reason": reason},
        )
    return True


def _cache_base() -> Optional[Path]:
    """Return the configured Hugging Face cache directory when available."""

    for env in ("TRANSFORMERS_CACHE", "HF_HOME"):
        loc = os.getenv(env)
        if loc:
            return Path(loc).expanduser()
    default = Path.home() / ".cache" / "huggingface"
    return default if default.exists() else None


def _cached_model_path(cache_dir: Path, model_id: str) -> Path:
    """Return the expected cache path for ``model_id`` within ``cache_dir``."""

    safe_name = model_id.replace("/", "--")
    return cache_dir / "hub" / f"models--{safe_name}"


def _resolve_local_snapshot(model_cache: Path) -> Optional[Path]:
    """Return the most recent cached snapshot for ``model_cache`` if available.

    Hugging Face caches model revisions in ``snapshots/<rev>``.  When the
    download was previously completed we can load the model directly from the
    snapshot directory without invoking the hub client which may otherwise block
    on network access.  The newest snapshot is preferred because older
    directories may be partially downloaded or left over from failed updates.
    """

    snapshots_root = model_cache / "snapshots"
    if not snapshots_root.exists():
        return None

    candidates: list[tuple[float, Path]] = []
    try:
        for entry in snapshots_root.iterdir():
            try:
                if not entry.is_dir():
                    continue
                stamp = entry.stat().st_mtime
            except FileNotFoundError:
                continue
            except Exception as exc:  # pragma: no cover - diagnostics only
                logger.debug("failed to stat snapshot %s: %s", entry, exc)
                continue
            candidates.append((stamp, entry))
    except FileNotFoundError:
        return None
    except Exception as exc:  # pragma: no cover - diagnostics only
        logger.debug("failed to scan snapshot directory %s: %s", snapshots_root, exc)
        return None

    metadata_candidates = ("modules.json", "sentence_bert_config.json")

    for _, path in sorted(candidates, key=lambda item: item[0], reverse=True):
        config = path / "config.json"
        try:
            if not config.exists():
                continue

            has_metadata = False
            for name in metadata_candidates:
                meta_path = path / name
                try:
                    if meta_path.exists():
                        has_metadata = True
                        break
                except FileNotFoundError:
                    continue
            if has_metadata:
                return path
        except FileNotFoundError:
            continue
        except Exception as exc:  # pragma: no cover - diagnostics only
            logger.debug("failed to inspect snapshot %s: %s", path, exc)
            continue
    return None


def embedder_cache_present() -> bool:
    """Return ``True`` when a cached embedder snapshot is available."""

    global _EMBEDDER
    if _EMBEDDER is not None:
        return True

    cache_dir = _cache_base()
    if cache_dir is None:
        return False

    model_cache = _cached_model_path(cache_dir, _MODEL_NAME)
    if not model_cache.exists():
        return False

    snapshot_path = _resolve_local_snapshot(model_cache)
    return snapshot_path is not None


def _is_meta_tensor_loading_error(exc: Exception) -> bool:
    """Return ``True`` when *exc* indicates the snapshot still has meta tensors."""

    message = str(exc)
    lowered = message.lower()
    return (
        "meta tensor" in lowered
        or "to_empty" in lowered
        or "no data" in lowered
        or "still contains meta tensors" in lowered
    )


def _module_contains_meta_tensors(module: "torch.nn.Module") -> bool:
    """Return ``True`` when *module* retains parameters or buffers on ``meta``."""

    if torch is None:
        return False

    def _iter_meta_flags(iterable: Iterable[object]) -> Iterable[bool]:
        for item in iterable:
            try:
                if getattr(item, "is_meta", False):
                    yield True
            except Exception:  # pragma: no cover - defensive only
                continue

    try:
        if any(_iter_meta_flags(module.parameters(recurse=True))):
            return True
    except Exception:  # pragma: no cover - defensive only
        pass

    try:
        if any(_iter_meta_flags(module.buffers(recurse=True))):
            return True
    except Exception:  # pragma: no cover - defensive only
        pass

    return False


def _materialise_sentence_transformer_device(
    instance: "SentenceTransformer",
    target_device: "str | object | None",
) -> "SentenceTransformer":
    """Move *instance* to *target_device* while handling meta tensor states."""

    if torch is None:
        raise RuntimeError("torch is required to materialise the sentence transformer")

    if target_device is None:
        return instance

    try:
        device = torch.device(target_device)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.debug("invalid embedder device %s: %s", target_device, exc)
        return instance

    if hasattr(instance, "to_empty"):
        try:
            result = instance.to_empty(device=device)
            if result is not None:
                instance = result
            if not _module_contains_meta_tensors(instance):
                return instance
        except Exception as exc:  # pragma: no cover - diagnostics only
            logger.debug(
                "failed to materialise sentence transformer with to_empty on %s: %s",
                device,
                exc,
            )

    module_to_empty = None
    if torch is not None:
        try:
            module_to_empty = getattr(torch.nn.Module, "to_empty", None)
        except Exception:  # pragma: no cover - defensive only
            module_to_empty = None

    if module_to_empty is not None:
        try:
            result = module_to_empty(instance, device=device)
            if result is not None:
                instance = result
            if not _module_contains_meta_tensors(instance):
                return instance
        except Exception as exc:  # pragma: no cover - diagnostics only
            logger.debug(
                "failed to materialise sentence transformer with to_empty on %s: %s",
                device,
                exc,
            )

    try:
        instance = instance.to(device)
    except Exception as exc:  # pragma: no cover - diagnostics only
        logger.debug("failed to move sentence transformer to %s: %s", device, exc)

    if _module_contains_meta_tensors(instance):
        raise RuntimeError("sentence transformer still contains meta tensors after materialisation")

    return instance


def initialise_sentence_transformer(
    identifier: str,
    *,
    device: "str | None" = None,
    force_meta_initialisation: bool = False,
    prefer_local: bool = False,
    **kwargs: object,
) -> "SentenceTransformer":
    """Construct a :class:`SentenceTransformer` handling meta tensor snapshots."""

    if SentenceTransformer is None:  # pragma: no cover - optional dependency missing
        raise RuntimeError("sentence-transformers is not available")

    init_kwargs = dict(kwargs)
    if prefer_local and "local_files_only" not in init_kwargs:
        init_kwargs["local_files_only"] = True
    if "local_files_only" not in init_kwargs:
        env_local_only = os.getenv("SANDBOX_EMBEDDER_LOCAL_ONLY", "").strip().lower()
        if env_local_only:
            init_kwargs["local_files_only"] = env_local_only not in {"0", "false", "off"}
    if device and "device" not in init_kwargs:
        init_kwargs["device"] = device

    target_device = init_kwargs.get("device") or SENTENCE_TRANSFORMER_DEVICE or "cpu"

    if force_meta_initialisation and target_device != "meta":
        meta_kwargs = dict(init_kwargs)
        meta_kwargs["device"] = "meta"
        model = SentenceTransformer(identifier, **meta_kwargs)
        return _materialise_sentence_transformer_device(model, target_device)

    try:
        return SentenceTransformer(identifier, **init_kwargs)
    except Exception as exc:
        if not _is_meta_tensor_loading_error(exc):
            raise

        logger.warning(
            "sentence transformer initialisation encountered meta tensors; retrying",
            extra={"model": identifier, "device": target_device},
        )

        meta_kwargs = dict(init_kwargs)
        meta_kwargs["device"] = "meta"
        model = SentenceTransformer(identifier, **meta_kwargs)
        return _materialise_sentence_transformer_device(model, target_device)


def _purge_corrupted_snapshot(snapshot_path: Path) -> None:
    """Remove a corrupted ``SentenceTransformer`` snapshot directory."""

    try:
        exists = snapshot_path.exists()
    except FileNotFoundError:
        return
    except Exception as exc:  # pragma: no cover - diagnostics only
        logger.debug(
            "failed to inspect corrupted snapshot %s: %s", snapshot_path, exc
        )
        return

    if not exists:
        return

    try:
        shutil.rmtree(snapshot_path)
    except FileNotFoundError:
        return
    except Exception as exc:  # pragma: no cover - diagnostics only
        logger.debug(
            "failed to remove corrupted snapshot %s: %s", snapshot_path, exc
        )
    else:
        logger.info(
            "removed corrupted sentence transformer snapshot",
            extra={"snapshot": str(snapshot_path)},
        )


def _cleanup_hf_locks(cache_dir: Path, *, focus: Path | None = None) -> None:
    """Remove stale Hugging Face lock files left behind by crashed downloads.

    ``focus`` allows callers to restrict the scan to a specific subtree (for
    example the cache directory of the embedder being loaded).  When provided it
    takes precedence over the broader ``cache_dir`` walk.  Older versions fell
    back to a full recursive scan of ``cache_dir`` even when the targeted
    directory was missing which can be extremely expensive on shared caches with
    many models.  The implementation now performs a shallow inspection in that
    scenario so startup does not stall while traversing unrelated directories.
    """

    if _HF_LOCK_CLEANUP_TIMEOUT == 0:
        return

    if not cache_dir.exists():
        return

    deadline: float | None = None
    if _HF_LOCK_CLEANUP_TIMEOUT > 0:
        deadline = time.monotonic() + _HF_LOCK_CLEANUP_TIMEOUT

    def _iter_lock_files(base: Path, *, recursive: bool) -> Iterable[Path]:
        if deadline is not None and time.monotonic() >= deadline:
            return []
        if not base.exists():
            return []

        if not recursive:
            try:
                iterator = base.iterdir()
            except FileNotFoundError:
                return []
            except Exception as exc:  # pragma: no cover - diagnostics only
                logger.debug("failed to list %s: %s", base, exc)
                return []
            result: list[Path] = []
            for entry in iterator:
                if deadline is not None and time.monotonic() >= deadline:
                    break
                try:
                    if entry.is_dir():
                        continue
                except FileNotFoundError:
                    continue
                except Exception as exc:  # pragma: no cover - diagnostics only
                    logger.debug("failed to inspect %s: %s", entry, exc)
                    continue
                name = entry.name
                if not name.endswith(".lock"):
                    continue
                if name == "menace-embedder.lock":
                    continue
                result.append(entry)
            return result

        try:
            for root, dirs, files in os.walk(base):
                if deadline is not None and time.monotonic() >= deadline:
                    dirs[:] = []
                    logger.debug(
                        "aborting huggingface lock cleanup after %.1fs", _HF_LOCK_CLEANUP_TIMEOUT
                    )
                    break
                for name in files:
                    if deadline is not None and time.monotonic() >= deadline:
                        break
                    if not name.endswith(".lock"):
                        continue
                    if name == "menace-embedder.lock":
                        continue
                    lock_path = Path(root) / name
                    try:
                        if lock_path.is_dir():
                            continue
                    except Exception:  # pragma: no cover - ignore racing filesystem errors
                        continue
                    yield lock_path
        except Exception as exc:  # pragma: no cover - diagnostics only
            logger.debug("failed to scan %s for locks: %s", base, exc)
        return []

    def _is_stale(path: Path) -> bool:
        stale = False
        if is_lock_stale is not None:
            try:
                stale = is_lock_stale(str(path), timeout=max(LOCK_TIMEOUT, 300))
            except Exception as exc:  # pragma: no cover - diagnostics only
                logger.debug("failed to check lock %s: %s", path, exc)
                stale = False
        if stale:
            return True
        try:
            return time.time() - path.stat().st_mtime > max(LOCK_TIMEOUT, 300)
        except FileNotFoundError:
            return False
        except Exception as exc:  # pragma: no cover - diagnostics only
            logger.debug("failed to stat lock file %s: %s", path, exc)
            return False

    targets: list[tuple[Path, bool]] = []
    if focus is not None and focus.exists():
        targets.append((focus, True))
    else:
        if focus is not None:
            parent = focus.parent if focus.parent != focus else cache_dir
            if parent.exists():
                targets.append((parent, False))
        targets.append((cache_dir, False))

    seen: set[Path] = set()
    for base, recursive in targets:
        if deadline is not None and time.monotonic() >= deadline:
            break
        if base in seen:
            continue
        seen.add(base)
        for lock_path in _iter_lock_files(base, recursive=recursive):
            if deadline is not None and time.monotonic() >= deadline:
                break
            if not _is_stale(lock_path):
                continue
            try:
                lock_path.unlink()
                logger.warning("removed stale huggingface lock: %s", lock_path)
            except FileNotFoundError:
                continue
            except Exception as exc:  # pragma: no cover - diagnostics only
                logger.debug(
                    "failed to remove stale huggingface lock %s: %s", lock_path, exc
                )


def _force_cleanup_embedder_locks() -> int:
    """Aggressively remove lock files within the embedder cache directory."""

    cache_dir = _cache_base()
    if cache_dir is None:
        return 0

    model_cache = _cached_model_path(cache_dir, _MODEL_NAME)
    if not model_cache.exists():
        return 0

    removed = 0

    deadline: float | None = None
    if _HF_LOCK_CLEANUP_TIMEOUT > 0:
        deadline = time.monotonic() + _HF_LOCK_CLEANUP_TIMEOUT

    with _FORCED_LOCK_CLEANUP_GUARD:
        try:
            for root, dirs, files in os.walk(model_cache):
                if deadline is not None and time.monotonic() >= deadline:
                    dirs[:] = []
                    logger.debug(
                        "aborting forced embedder lock cleanup after %.1fs",
                        _HF_LOCK_CLEANUP_TIMEOUT,
                    )
                    break

                for name in files:
                    if deadline is not None and time.monotonic() >= deadline:
                        dirs[:] = []
                        break
                    if not name.endswith(".lock"):
                        continue
                    if name == "menace-embedder.lock":
                        continue

                    lock_path = Path(root) / name
                    try:
                        if lock_path.is_dir():
                            continue
                    except Exception:  # pragma: no cover - ignore racing filesystem errors
                        continue

                    try:
                        lock_path.unlink()
                        removed += 1
                    except FileNotFoundError:
                        continue
                    except Exception as exc:  # pragma: no cover - diagnostics only
                        logger.debug(
                            "failed to remove embedder cache lock %s: %s", lock_path, exc
                        )

                if deadline is not None and time.monotonic() >= deadline:
                    dirs[:] = []
                    logger.debug(
                        "aborting forced embedder lock cleanup after %.1fs",
                        _HF_LOCK_CLEANUP_TIMEOUT,
                    )
                    break
        except FileNotFoundError:
            return removed
        except Exception as exc:  # pragma: no cover - diagnostics only
            logger.debug(
                "failed to scan embedder cache for locks: %s", exc,
                extra={"cache": str(model_cache)},
            )
            return removed

    if removed:
        logger.warning(
            "force removed %d stale huggingface locks from %s",
            removed,
            model_cache,
        )
    return removed


def _embedder_lock() -> SandboxLockType | None:
    """Return a process-wide file lock guarding embedder initialisation."""

    global _EMBEDDER_LOCK
    if SandboxLock is None:
        return None
    if _EMBEDDER_LOCK is not None:
        return _EMBEDDER_LOCK

    base = _cache_base()
    if base is None:
        base = Path.home() / ".cache" / "huggingface"
    try:
        base.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # pragma: no cover - best effort logging
        logger.debug("failed to prepare embedder lock directory: %s", exc)
        return None

    lock_path = base / "menace-embedder.lock"
    _EMBEDDER_LOCK = SandboxLock(str(lock_path))
    return _EMBEDDER_LOCK


def _cleanup_stale_embedder_lock(lock_path: str) -> bool:
    """Attempt to remove a stale embedder lock file."""

    if not lock_path:
        return False

    try:
        path = Path(lock_path)
    except Exception:  # pragma: no cover - invalid paths are ignored
        return False

    stale = False
    try:
        if is_lock_stale is not None:
            stale = is_lock_stale(lock_path, timeout=max(LOCK_TIMEOUT, 300))
    except Exception as exc:  # pragma: no cover - diagnostics only
        logger.debug("failed to inspect embedder lock %s: %s", lock_path, exc)
        stale = False

    if not stale:
        try:
            stale = time.time() - path.stat().st_mtime > max(LOCK_TIMEOUT, 300)
        except FileNotFoundError:
            return False
        except Exception as exc:  # pragma: no cover - diagnostics only
            logger.debug("failed to stat embedder lock %s: %s", lock_path, exc)
            return False

    if not stale:
        return False

    try:
        path.unlink()
    except FileNotFoundError:
        return False
    except Exception as exc:  # pragma: no cover - diagnostics only
        logger.debug("failed to remove stale embedder lock %s: %s", lock_path, exc)
        return False

    logger.warning("removed stale embedder lock: %s", lock_path)
    return True


def _identify_embedder_requester() -> str | None:
    """Return a concise description of the caller requesting the embedder."""

    try:
        stack = inspect.stack()
    except Exception:
        return None
    try:
        for frame_info in stack[2:]:
            module = inspect.getmodule(frame_info.frame)
            mod_name = module.__name__ if module is not None else None
            if mod_name is None:
                continue
            if mod_name.startswith(__name__):
                continue
            if mod_name.startswith("threading") or mod_name.startswith("logging"):
                continue
            return f"{mod_name}.{frame_info.function}"
        if len(stack) > 2:
            frame_info = stack[2]
            module = inspect.getmodule(frame_info.frame)
            mod_name = module.__name__ if module is not None else None
            return f"{mod_name}.{frame_info.function}" if mod_name else frame_info.function
        return None
    finally:
        del stack


def _embedder_disabled(stop_event: threading.Event | None = None) -> bool:
    return _EMBEDDER_DISABLED or _embedder_stop_requested(stop_event)


def _ensure_embedder_thread_locked(
    stop_event: threading.Event | None = None,
    *,
    load_context: dict[str, object] | None = None,
) -> threading.Event:
    """Ensure the background embedder initialisation thread is running."""

    global _EMBEDDER_INIT_THREAD, _EMBEDDER_TIMEOUT_LOGGED, _EMBEDDER_TIMEOUT_REACHED

    if _embedder_disabled(stop_event):
        _EMBEDDER_INIT_EVENT.set()
        _trace(
            "thread.skip",
            reason="embedder_disabled",
            disabled=_EMBEDDER_DISABLED,
            stop_requested=_embedder_stop_requested(stop_event),
        )
        return _EMBEDDER_INIT_EVENT

    if _EMBEDDER_INIT_THREAD is not None and _EMBEDDER_INIT_THREAD.is_alive():
        _trace("thread.exists", thread_name=_EMBEDDER_INIT_THREAD.name)
        return _EMBEDDER_INIT_EVENT

    _EMBEDDER_INIT_EVENT.clear()
    _EMBEDDER_TIMEOUT_LOGGED = False
    _EMBEDDER_TIMEOUT_REACHED = False
    global _FALLBACK_ANNOUNCED
    _FALLBACK_ANNOUNCED = False

    def _initialise() -> None:
        global _EMBEDDER, _EMBEDDER_TIMEOUT_LOGGED
        start = time.perf_counter()
        _trace("thread.loader.start")
        logger.info("embedder initialisation thread started", extra={"model": _MODEL_NAME})
        try:
            if _embedder_stop_requested():
                logger.info(
                    "embedder initialisation cancelled before start",
                    extra={"model": _MODEL_NAME},
                )
                _trace("thread.loader.cancelled", reason="pre-start")
                return
            load_kwargs = load_context or {}
            result = _load_embedder(stop_event=_EMBEDDER_STOP_EVENT, **load_kwargs)
            if result is not None and not _embedder_stop_requested():
                _EMBEDDER = result
        except Exception:  # pragma: no cover - defensive logging
            logger.debug("sentence transformer initialisation raised", exc_info=True)
            result = None
        finally:
            duration = time.perf_counter() - start
            logger.info(
                "embedder initialisation thread finished",
                extra={
                    "model": _MODEL_NAME,
                    "duration": round(duration, 3),
                    "success": _EMBEDDER is not None,
                },
            )
            _trace(
                "thread.loader.finished",
                duration=round(duration, 3),
                success=_EMBEDDER is not None,
            )
            if _EMBEDDER is not None and _EMBEDDER_TIMEOUT_LOGGED:
                logger.info("sentence transformer became available after initial timeout")
            _EMBEDDER_INIT_EVENT.set()

    _EMBEDDER_INIT_THREAD = threading.Thread(
        target=_initialise,
        name="menace-embedder-init",
        daemon=True,
    )
    _EMBEDDER_INIT_THREAD.start()
    _trace("thread.spawned", thread_name=_EMBEDDER_INIT_THREAD.name)
    return _EMBEDDER_INIT_EVENT


def _initialise_embedder_with_timeout(
    timeout_override: float | None = None,
    *,
    suppress_timeout_log: bool = False,
    requester: str | None = None,
    stop_event: threading.Event | None = None,
    fallback_on_timeout: bool = False,
    max_wait_override: float | None = None,
    allow_background_wait: bool = False,
    load_budget_deadline: float | None = None,
) -> SentenceTransformer | Any | None:
    """Initialise the shared embedder without blocking indefinitely.

    ``timeout_override`` allows callers to shorten the wait period without
    affecting the global timeout configuration.  When an override is used the
    function returns ``None`` once the shorter timeout elapses while keeping the
    background initialisation thread alive.
    """

    global _EMBEDDER_TIMEOUT_LOGGED, _EMBEDDER_SOFT_WAIT_LOGGED, _EMBEDDER_TIMEOUT_REACHED, _EMBEDDER_STOP_EVENT

    with _EMBEDDER_THREAD_LOCK:
        if _EMBEDDER is not None:
            return _EMBEDDER
        if stop_event is None and _EMBEDDER_STOP_EVENT is not None and _EMBEDDER_STOP_EVENT.is_set():
            _EMBEDDER_STOP_EVENT = None
        if stop_event is not None:
            _EMBEDDER_STOP_EVENT = stop_event
        load_context = {
            "budget_deadline": load_budget_deadline,
            "bootstrap_mode": allow_background_wait,
            "requester": requester,
        }
        event = _ensure_embedder_thread_locked(stop_event, load_context=load_context)

    if _embedder_disabled(stop_event):
        logger.info(
            "embedder initialisation skipped because embedder is disabled",  # pragma: no cover - logging only
            extra={"model": _MODEL_NAME},
        )
        _EMBEDDER_TIMEOUT_REACHED = True
        _EMBEDDER_INIT_EVENT.set()
        return None

    if _embedder_stop_requested(stop_event):
        _cancel_embedder_initialisation(stop_event, reason="stop_requested")
        if _activate_bundled_fallback("bootstrap_cancelled") and _EMBEDDER is not None:
            return _EMBEDDER
        placeholder = _build_stub_embedder()
        setattr(placeholder, "_placeholder_reason", "stop_requested")
        return placeholder

    if _EMBEDDER_TIMEOUT_REACHED and not event.is_set():
        thread = _EMBEDDER_INIT_THREAD
        alive = thread.is_alive() if thread is not None else False
        if suppress_timeout_log:
            logger.debug(
                "skipping embedder wait after previous timeout",
                extra={"model": _MODEL_NAME, "thread_alive": alive},
            )
        else:
            logger.info(
                "embedder initialisation previously timed out; returning cached failure",
                extra={"model": _MODEL_NAME, "thread_alive": alive},
            )
        return _EMBEDDER

    global _EMBEDDER_WAIT_CAPPED

    requested_timeout = _EMBEDDER_INIT_TIMEOUT
    if timeout_override is not None:
        requested_timeout = min(requested_timeout, max(0.0, timeout_override))

    max_wait = (
        _MAX_EMBEDDER_WAIT
        if max_wait_override is None
        else min(_MAX_EMBEDDER_WAIT, max_wait_override)
    )
    wait_cap = max(0.0, min(requested_timeout, max_wait))
    wait_limit = wait_cap
    soft_clamped = False
    if _SOFT_EMBEDDER_WAIT >= 0:
        soft_cap = _SOFT_EMBEDDER_WAIT
        if timeout_override is not None:
            soft_cap = min(soft_cap, max(0.0, timeout_override))
        wait_limit = min(wait_limit, soft_cap)
        soft_clamped = wait_limit < wait_cap

    skip_wait = False
    log_wait = True
    if requester:
        with _EMBEDDER_REQUESTER_LOCK:
            skip_wait = requester in _EMBEDDER_REQUESTER_TIMEOUTS
            if requester in _EMBEDDER_REQUESTER_LOGGED:
                log_wait = False
            else:
                _EMBEDDER_REQUESTER_LOGGED.add(requester)
    if skip_wait:
        wait_limit = 0.0

    if (
        timeout_override is None
        and not _EMBEDDER_WAIT_CAPPED
        and (wait_limit < _EMBEDDER_INIT_TIMEOUT or soft_clamped)
        and not _EMBEDDER_TIMEOUT_LOGGED
    ):
        _EMBEDDER_WAIT_CAPPED = True
        if soft_clamped and not _EMBEDDER_SOFT_WAIT_LOGGED:
            _EMBEDDER_SOFT_WAIT_LOGGED = True
            logger.warning(
                "capping embedder initialisation wait to %.0fs due to EMBEDDER_INIT_SOFT_WAIT=%.0fs (requested %.0fs)",
                wait_limit,
                _SOFT_EMBEDDER_WAIT,
                _EMBEDDER_INIT_TIMEOUT,
            )
        else:
            logger.warning(
                "capping embedder initialisation wait to %.0fs (requested %.0fs)",
                wait_limit,
                _EMBEDDER_INIT_TIMEOUT,
            )

    wait_time = wait_limit
    if _EMBEDDER_TIMEOUT_LOGGED and not suppress_timeout_log:
        wait_time = 0.0

    if wait_time > 0:
        if requester and log_wait:
            logger.debug(
                "waiting up to %.1fs for embedder initialisation (requested by %s)",
                wait_time,
                requester,
                extra={
                    "model": _MODEL_NAME,
                    "wait_time": round(wait_time, 3),
                    "requested_timeout": round(requested_timeout, 3),
                    "max_wait": round(max_wait, 3),
                    "soft_wait": round(_SOFT_EMBEDDER_WAIT, 3),
                },
            )
            _trace(
                "wait.start",
                requester=requester,
                wait_time=round(wait_time, 3),
                requested_timeout=round(requested_timeout, 3),
            )
        elif not requester:
            logger.debug(
                "waiting up to %.1fs for embedder initialisation",
                wait_time,
                extra={
                    "model": _MODEL_NAME,
                    "wait_time": round(wait_time, 3),
                    "requested_timeout": round(requested_timeout, 3),
                    "max_wait": round(_MAX_EMBEDDER_WAIT, 3),
                    "soft_wait": round(_SOFT_EMBEDDER_WAIT, 3),
                },
            )
            _trace(
                "wait.start",
                requester=None,
                wait_time=round(wait_time, 3),
                requested_timeout=round(requested_timeout, 3),
            )
    finished = False
    waited = 0.0
    remaining_wait = wait_time
    if wait_time <= 0:
        finished = bool(getattr(event, "is_set", lambda: False)())
    else:
        deadline = time.perf_counter() + wait_time
        heartbeat_interval = 5.0
        if wait_time <= heartbeat_interval:
            start_wait = time.perf_counter()
            finished = event.wait(wait_time)
            waited = time.perf_counter() - start_wait
            if waited > wait_time:
                waited = wait_time
            remaining = max(0.0, wait_time - waited)
            remaining_wait = max(0.0, deadline - time.perf_counter())
        else:
            next_heartbeat = time.perf_counter() + heartbeat_interval
            remaining = wait_time
            while remaining > 0:
                if remaining <= 1e-6:
                    break
                slice_wait = 1.0 if remaining > 1.0 else remaining
                start_slice = time.perf_counter()
                if event.wait(slice_wait):
                    finished = True
                    break
                elapsed = time.perf_counter() - start_slice
                if _embedder_stop_requested(stop_event):
                    finished = False
                    break
                remaining = max(0.0, deadline - time.perf_counter())
                waited = wait_time - max(0.0, remaining)
                remaining_wait = max(0.0, remaining)
                if elapsed < slice_wait and remaining > 0:
                    time.sleep(min(0.1, remaining))
                if time.perf_counter() >= next_heartbeat:
                    next_heartbeat = time.perf_counter() + heartbeat_interval
                    logger.debug(
                        "sentence transformer initialisation still pending",
                        extra={
                            "model": _MODEL_NAME,
                            "seconds_remaining": round(remaining_wait, 3),
                            "requester": requester,
                        },
                    )
                    _trace(
                        "wait.heartbeat",
                        requester=requester,
                        seconds_remaining=round(remaining_wait, 3),
                        waited=round(waited, 3),
                    )
            else:
                remaining = max(0.0, deadline - time.perf_counter())
            waited = max(0.0, wait_time - max(0.0, remaining))
            remaining_wait = max(0.0, remaining)
        if not finished:
            finished = bool(getattr(event, "is_set", lambda: False)())
    if _embedder_stop_requested(stop_event):
        _cancel_embedder_initialisation(stop_event, reason="stop_requested")
        if _activate_bundled_fallback("bootstrap_cancelled") and _EMBEDDER is not None:
            return _EMBEDDER
        placeholder = _build_stub_embedder()
        setattr(placeholder, "_placeholder_reason", "stop_requested")
        return placeholder
    if finished:
        _EMBEDDER_TIMEOUT_LOGGED = False
        _EMBEDDER_TIMEOUT_REACHED = False
        if requester:
            with _EMBEDDER_REQUESTER_LOCK:
                _EMBEDDER_REQUESTER_TIMEOUTS.discard(requester)
        _trace(
            "wait.finished",
            requester=requester,
            wait_time=round(wait_time, 3),
            waited=round(waited, 3),
        )
        return _EMBEDDER

    _dump_embedder_thread("wait_timeout", waited or wait_time or 0.0)
    cleaned = False
    if not (fallback_on_timeout and allow_background_wait):
        _cancel_embedder_initialisation(stop_event, reason="wait_timeout")
        cleaned = _force_cleanup_embedder_locks()
        if cleaned:
            _trace(
                "wait.force_cleanup",
                requester=requester,
                removed=cleaned,
            )
    else:
        _trace(
            "wait.background_continues",
            requester=requester,
            waited=round(waited or wait_time or 0.0, 3),
        )

    placeholder = None

    if _EMBEDDER is None and _activate_bundled_fallback("timeout"):
        _EMBEDDER_TIMEOUT_LOGGED = False
        _EMBEDDER_TIMEOUT_REACHED = False
        if requester:
            with _EMBEDDER_REQUESTER_LOCK:
                _EMBEDDER_REQUESTER_TIMEOUTS.add(requester)
        logger.info(
            "embedder initialisation deferred after %.1fs; using fallback stub",
            waited or wait_time,
            extra={"model": _MODEL_NAME, "requester": requester},
        )
        _trace(
            "wait.timeout.fallback",
            requester=requester,
            waited=round(waited or wait_time or 0.0, 3),
        )
        return _EMBEDDER

    if fallback_on_timeout:
        placeholder = _build_stub_embedder()
        setattr(placeholder, "_placeholder_reason", "timeout")
        logger.info(
            "embedder initialisation exceeded %.1fs; returning placeholder embedder",
            waited or wait_time,
            extra={"model": _MODEL_NAME, "requester": requester},
        )
        _trace(
            "wait.timeout.placeholder",
            requester=requester,
            waited=round(waited or wait_time or 0.0, 3),
        )

    if suppress_timeout_log:
        if wait_time > 0:
            if requester:
                logger.debug(
                    "sentence transformer initialisation still pending after %.0fs (requested by %s)",
                    waited or wait_time,
                    requester,
                    extra={"model": _MODEL_NAME},
                )
            else:
                logger.debug(
                    "sentence transformer initialisation still pending after %.0fs",
                    waited or wait_time,
                    extra={"model": _MODEL_NAME},
                )
            _trace(
                "wait.pending",
                requester=requester,
                wait_time=round(waited or wait_time, 3),
                suppress_timeout=True,
            )
        _EMBEDDER_TIMEOUT_REACHED = True
        if requester:
            with _EMBEDDER_REQUESTER_LOCK:
                _EMBEDDER_REQUESTER_TIMEOUTS.add(requester)
        return placeholder or _EMBEDDER

    if not _EMBEDDER_TIMEOUT_LOGGED:
        _EMBEDDER_TIMEOUT_LOGGED = True
        _EMBEDDER_TIMEOUT_REACHED = True
        thread = _EMBEDDER_INIT_THREAD
        alive = thread.is_alive() if thread is not None else False
        if requester:
            logger.error(
                "sentence transformer initialisation exceeded %.0fs; continuing without embeddings (requested by %s)",
                _EMBEDDER_INIT_TIMEOUT,
                requester,
                extra={"model": _MODEL_NAME, "thread_alive": alive},
            )
            _trace(
                "wait.timeout",
                requester=requester,
                wait_time=round(waited or wait_time, 3),
                thread_alive=alive,
            )
        else:
            logger.error(
                "sentence transformer initialisation exceeded %.0fs; continuing without embeddings",
                _EMBEDDER_INIT_TIMEOUT,
                extra={"model": _MODEL_NAME, "thread_alive": alive},
            )
            _trace(
                "wait.timeout",
                requester=None,
                wait_time=round(waited or wait_time, 3),
                thread_alive=alive,
            )
        if requester:
            with _EMBEDDER_REQUESTER_LOCK:
                _EMBEDDER_REQUESTER_TIMEOUTS.add(requester)
        _cancel_embedder_initialisation(stop_event, reason="timeout_logged")
    return placeholder or _EMBEDDER


def _cancel_embedder_initialisation(
    stop_event: threading.Event | None, *, reason: str, join_timeout: float = 2.0
) -> None:
    global _EMBEDDER_INIT_THREAD
    if stop_event is None:
        return
    stop_event.set()
    _EMBEDDER_INIT_EVENT.set()
    thread = _EMBEDDER_INIT_THREAD
    if thread is None:
        return
    thread.join(join_timeout)
    if thread.is_alive():
        logger.debug(
            "embedder initialisation thread still running after cancellation",
            extra={"model": _MODEL_NAME, "reason": reason, "timeout": join_timeout},
        )
    else:
        logger.info(
            "embedder initialisation thread cancelled",
            extra={"model": _MODEL_NAME, "reason": reason},
        )
    _EMBEDDER_INIT_THREAD = None


def disable_embedder(
    *,
    reason: str = "disabled",
    stop_event: threading.Event | None = None,
    join_timeout: float = 2.0,
) -> None:
    """Disable embedder initialisation for the current process."""

    global _EMBEDDER_DISABLED, _EMBEDDER_STOP_EVENT
    _EMBEDDER_DISABLED = True
    if stop_event is not None:
        _EMBEDDER_STOP_EVENT = stop_event
    _cancel_embedder_initialisation(
        stop_event or _EMBEDDER_STOP_EVENT,
        reason=reason,
        join_timeout=join_timeout,
    )
    _EMBEDDER_INIT_EVENT.set()
    logger.info(
        "embedder initialisation disabled",  # pragma: no cover - logging only
        extra={"model": _MODEL_NAME, "reason": reason},
    )


def _embedder_stop_requested(stop_event: threading.Event | None = None) -> bool:
    event = stop_event if stop_event is not None else _EMBEDDER_STOP_EVENT
    return bool(event and event.is_set())


def _load_embedder(
    stop_event: threading.Event | None = None,
    *,
    budget_deadline: float | None = None,
    bootstrap_mode: bool = False,
    requester: str | None = None,
) -> SentenceTransformer | None:
    """Load the shared ``SentenceTransformer`` instance with offline fallbacks."""

    global model

    budget_placeholder = _guard_bootstrap_budget(
        budget_deadline=budget_deadline,
        stop_event=stop_event,
        bootstrap_mode=bootstrap_mode,
        requester=requester,
        stage="prefetch",
    )
    if budget_placeholder is not None:
        return cast("SentenceTransformer", budget_placeholder)

    if _embedder_stop_requested(stop_event):
        logger.info("embedder initialisation cancelled", extra={"model": _MODEL_NAME})
        _trace("load.cancelled")
        return None

    _trace(
        "load.sync.start",
        bootstrap_mode=bootstrap_mode,
        requester=requester,
    )

    if SentenceTransformer is None:  # pragma: no cover - optional dependency missing
        fallback = _load_bundled_embedder()
        if fallback is not None:
            return cast("SentenceTransformer", fallback)
        return None

    if model is not None:
        logger.info(
            "using preloaded sentence transformer",
            extra={"model": _MODEL_NAME},
        )
        _trace("load.preloaded")
        return model

    cache_dir = _cache_base()
    local_kwargs: dict[str, object] = {}
    bootstrap_force_local = False
    prefer_local = bootstrap_mode
    start = time.perf_counter()
    _trace(
        "load.start",
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    logger.info(
        "starting sentence transformer initialisation",
        extra={
            "model": _MODEL_NAME,
            "cache_dir": str(cache_dir) if cache_dir is not None else None,
        },
    )
    budget_placeholder = _guard_bootstrap_budget(
        budget_deadline=budget_deadline,
        stop_event=stop_event,
        bootstrap_mode=bootstrap_mode,
        requester=requester,
        stage="cache_preflight",
    )
    if budget_placeholder is not None:
        return cast("SentenceTransformer", budget_placeholder)
    cache_present = False
    if cache_dir is not None:
        model_cache = _cached_model_path(cache_dir, _MODEL_NAME)
        cache_present = model_cache.exists()
        focus_dir = model_cache if cache_present else model_cache.parent
        try:
            _cleanup_hf_locks(cache_dir, focus=focus_dir)
        except TypeError:
            # Tests stub ``_cleanup_hf_locks`` with a positional-only lambda.
            # Fallback to the legacy calling convention to keep compatibility.
            _cleanup_hf_locks(cache_dir)
        local_kwargs["cache_folder"] = str(cache_dir)
        snapshot_path = _resolve_local_snapshot(model_cache) if cache_present else None
        if snapshot_path is not None:
            prefer_local = True
            logger.info(
                "loading sentence transformer snapshot from cache",
                extra={
                    "model": _MODEL_NAME,
                    "snapshot": str(snapshot_path),
                },
            )
            _trace(
                "load.snapshot.detected",
                snapshot=str(snapshot_path),
            )
            budget_placeholder = _guard_bootstrap_budget(
                budget_deadline=budget_deadline,
                stop_event=stop_event,
                bootstrap_mode=bootstrap_mode,
                requester=requester,
                stage="snapshot_detected",
            )
            if budget_placeholder is not None:
                return cast("SentenceTransformer", budget_placeholder)

        cache_has_refs = False
        cache_has_blobs = False
        if model_cache.exists():
            refs_dir = model_cache / "refs"
            try:
                cache_has_refs = refs_dir.exists() and any(refs_dir.iterdir())
            except FileNotFoundError:
                cache_has_refs = False
            except Exception as exc:  # pragma: no cover - diagnostics only
                logger.debug(
                    "failed to inspect embedder refs directory %s: %s",
                    refs_dir,
                    exc,
                )
            blobs_dir = model_cache / "blobs"
            try:
                if blobs_dir.exists():
                    next(blobs_dir.iterdir())
                    cache_has_blobs = True
            except StopIteration:
                cache_has_blobs = False
            except FileNotFoundError:
                cache_has_blobs = False
            except Exception as exc:  # pragma: no cover - diagnostics only
                logger.debug(
                    "failed to inspect embedder blobs directory %s: %s",
                    blobs_dir,
                    exc,
                )

    offline_env = os.environ.get("HF_HUB_OFFLINE") or os.environ.get("TRANSFORMERS_OFFLINE")
    force_local = os.environ.get("SANDBOX_FORCE_LOCAL_EMBEDDER", "").lower() not in {
        "",
        "0",
        "false",
    }
    if bootstrap_mode:
        bootstrap_force_local = (
            os.environ.get("SANDBOX_BOOTSTRAP_EMBEDDER_REMOTE", "")
            .strip()
            .lower()
            in {"", "0", "false", "off"}
        )
    if bootstrap_mode or cache_present:
        local_kwargs.setdefault("local_files_only", True)
    if bootstrap_force_local:
        local_kwargs.setdefault("local_files_only", True)
        _trace("load.bootstrap.local_only")
    prefer_cached = snapshot_path is not None
    partial_cache = cache_has_refs or cache_has_blobs
    if offline_env or force_local or prefer_cached:
        local_kwargs["local_files_only"] = True
        _trace(
            "load.local_mode",
            offline=bool(offline_env),
            force_local=bool(force_local),
            snapshot=bool(snapshot_path),
            prefer_cached=prefer_cached,
            partial_cache=partial_cache,
        )
        if prefer_cached and not offline_env and not force_local:
            logger.info(
                "using cached sentence transformer artefacts without hub access",
                extra={
                    "model": _MODEL_NAME,
                    "has_snapshot": bool(snapshot_path),
                    "has_refs": cache_has_refs,
                    "has_blobs": cache_has_blobs,
                },
            )
        elif partial_cache:
            _trace(
                "load.partial_cache.detected",
                has_refs=cache_has_refs,
                has_blobs=cache_has_blobs,
            )

        if snapshot_path is not None:
            try:
                logger.info(
                    "loading sentence transformer from cached snapshot",
                    extra={"model": _MODEL_NAME, "snapshot": str(snapshot_path)},
                )
                snapshot_kwargs: dict[str, object] = {"local_files_only": True}
                device = SENTENCE_TRANSFORMER_DEVICE
                if device:
                    snapshot_kwargs.setdefault("device", device)
                model = initialise_sentence_transformer(
                    str(snapshot_path),
                    force_meta_initialisation=True,
                    prefer_local=True,
                    **snapshot_kwargs,
                )
                duration = time.perf_counter() - start
                logger.info(
                    "loaded sentence transformer from snapshot",
                    extra={
                        "model": _MODEL_NAME,
                        "snapshot": str(snapshot_path),
                        "duration": round(duration, 3),
                    },
                )
                _trace(
                    "load.snapshot.success",
                    duration=round(duration, 3),
                )
                return model
            except Exception as exc:  # pragma: no cover - diagnostics only
                corrupted = _is_meta_tensor_loading_error(exc)
                logger.warning(
                    "failed to load sentence transformer from cached snapshot %s: %s",
                    snapshot_path,
                    exc,
                )
                _trace(
                    "load.snapshot.error",
                    error=str(exc),
                    exc_type=type(exc).__name__,
                )
                if corrupted:
                    _trace(
                        "load.snapshot.corrupted",
                        snapshot=str(snapshot_path),
                    )
                    _purge_corrupted_snapshot(snapshot_path)
                    local_kwargs.pop("local_files_only", None)

    if (prefer_local or cache_present) and "local_files_only" not in local_kwargs:
        local_kwargs["local_files_only"] = True

    if local_kwargs.get("local_files_only") and snapshot_path is None:
        _trace(
            "load.local_only.fallback",
            cache_present=cache_present,
            snapshot_available=bool(snapshot_path),
        )
        logger.warning(
            "local-only embedder requested but no cached snapshot is available; using bundled fallback with degraded embeddings",
            extra={"model": _MODEL_NAME},
        )
        fallback = _load_bundled_embedder()
        return cast("SentenceTransformer | None", fallback)

    token = os.getenv("HUGGINGFACE_API_TOKEN")
    if token:
        # The transformers stack honours these environment variables directly and
        # avoids the interactive ``huggingface_hub.login`` flow that can hang in
        # restricted environments.
        os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", token)
        os.environ.setdefault("HF_HUB_TOKEN", token)

    if _embedder_stop_requested(stop_event):
        logger.info(
            "embedder initialisation cancelled before hub load",
            extra={"model": _MODEL_NAME},
        )
        _trace("load.cancelled", stage="hub")
        return None

    try:
        logger.info(
            "loading sentence transformer via hub",
            extra={
                "model": _MODEL_NAME,
                "local_files_only": local_kwargs.get("local_files_only", False),
            },
        )
        _trace(
            "load.hub.start",
            local_only=bool(local_kwargs.get("local_files_only", False)),
        )
        _ensure_hf_timeouts()
        budget_placeholder = _guard_bootstrap_budget(
            budget_deadline=budget_deadline,
            stop_event=stop_event,
            bootstrap_mode=bootstrap_mode,
            requester=requester,
            stage="hub_load",
        )
        if budget_placeholder is not None:
            return cast("SentenceTransformer", budget_placeholder)
        if SENTENCE_TRANSFORMER_DEVICE and "device" not in local_kwargs:
            local_kwargs["device"] = SENTENCE_TRANSFORMER_DEVICE
        model = initialise_sentence_transformer(
            _MODEL_ID, prefer_local=prefer_local, **local_kwargs
        )
        duration = time.perf_counter() - start
        logger.info(
            "loaded sentence transformer",
            extra={"model": _MODEL_NAME, "duration": round(duration, 3)},
        )
        _trace(
            "load.hub.success",
            duration=round(duration, 3),
        )
        return model
    except Exception as exc:
        if local_kwargs.pop("local_files_only", None):
            if offline_env or bootstrap_force_local or force_local:
                logger.warning(
                    "sentence transformer initialisation failed in local-only mode: %s; using degraded fallback",
                    exc,
                )
                _trace(
                    "load.hub.error",
                    error=str(exc),
                    exc_type=type(exc).__name__,
                    local_only=True,
                )
                fallback = _load_bundled_embedder()
                if fallback is not None:
                    logger.warning(
                        "falling back to bundled embedder after local-only failure; embeddings may be degraded",
                        extra={"model": _MODEL_NAME},
                    )
                    return cast("SentenceTransformer", fallback)
                return None
            try:
                logger.info(
                    "retrying sentence transformer load with hub access",
                    extra={"model": _MODEL_NAME},
                )
                _trace("load.hub.retry")
                _ensure_hf_timeouts()
                model = initialise_sentence_transformer(
                    _MODEL_ID, prefer_local=prefer_local, **local_kwargs
                )
                duration = time.perf_counter() - start
                logger.info(
                    "loaded sentence transformer after retry",
                    extra={"model": _MODEL_NAME, "duration": round(duration, 3)},
                )
                _trace(
                    "load.hub.retry.success",
                    duration=round(duration, 3),
                )
                return model
            except Exception:
                logger.warning("failed to initialise sentence transformer: %s", exc)
                _trace(
                    "load.hub.retry.error",
                    error=str(exc),
                    exc_type=type(exc).__name__,
                )
                fallback = _load_bundled_embedder()
                if fallback is not None:
                    logger.warning(
                        "falling back to bundled embedder after retry; embeddings may be degraded",
                        extra={"model": _MODEL_NAME},
                    )
                    return cast("SentenceTransformer", fallback)
                return None
        logger.warning("failed to initialise sentence transformer: %s", exc)
        _trace(
            "load.hub.error",
            error=str(exc),
            exc_type=type(exc).__name__,
            local_only=False,
        )
        fallback = _load_bundled_embedder()
        if fallback is not None:
            logger.warning(
                "using bundled sentence transformer fallback after hub failure; embeddings may be degraded",
                extra={"model": _MODEL_NAME, "duration": round(time.perf_counter() - start, 3)},
            )
            _trace("load.fallback", duration=round(time.perf_counter() - start, 3))
            return cast("SentenceTransformer", fallback)
        logger.error(
            "sentence transformer initialisation failed with no fallback available",
            extra={"model": _MODEL_NAME},
        )
        _trace("load.error", fallback_available=False)
        return None


def get_embedder(
    timeout: float | None = None,
    *,
    stop_event: threading.Event | None = None,
    bootstrap_timeout: float | None = None,
    bootstrap_mode: bool = False,
) -> SentenceTransformer | None:
    """Return a lazily-instantiated shared :class:`SentenceTransformer`.

    The model is created on first use and reused for subsequent calls.  Errors
    during initialisation result in ``None`` being cached to avoid repeated
    expensive failures.  When ``timeout`` is provided the caller-specific wait
    duration is capped to that value, allowing services that run during startup
    to avoid blocking for the full global timeout.
    """
    global _EMBEDDER
    call_start = time.perf_counter()
    if _EMBEDDER is not None:
        return _EMBEDDER

    if bootstrap_mode and _EMBEDDER_BOOTSTRAP_DEFERRED and _EMBEDDER_BOOTSTRAP_PLACEHOLDER is not None:
        logger.debug(
            "returning memoized bootstrap embedder placeholder",
            extra={"model": _MODEL_NAME},
        )
        return _EMBEDDER_BOOTSTRAP_PLACEHOLDER

    requester = _identify_embedder_requester()
    wait_override = timeout
    bootstrap_wait_cap: float | None = None
    bootstrap_deadline: float | None = None
    max_wait_override: float | None = None
    budget_deadline: float | None = None
    if bootstrap_mode or bootstrap_timeout is not None:
        cap_budget = bootstrap_timeout if bootstrap_timeout is not None else timeout
        bootstrap_wait_cap = apply_bootstrap_timeout_caps(cap_budget)
        bootstrap_deadline = bootstrap_wait_cap
        max_wait_override = bootstrap_wait_cap
        if wait_override is None and bootstrap_wait_cap is not None:
            wait_override = bootstrap_wait_cap
        elif bootstrap_wait_cap is not None and wait_override is not None:
            wait_override = min(wait_override, bootstrap_wait_cap)
        elif bootstrap_wait_cap is not None:
            wait_override = bootstrap_wait_cap
        if bootstrap_wait_cap is not None and bootstrap_wait_cap >= 0:
            budget_deadline = call_start + bootstrap_wait_cap

    if bootstrap_deadline is not None and stop_event is None:
        stop_event = threading.Event()

    if bootstrap_deadline is not None and bootstrap_deadline > 0 and not bootstrap_mode:
        def _budget_guard(evt: threading.Event, budget: float) -> None:
            deadline = time.perf_counter() + budget
            while not evt.is_set():
                if time.perf_counter() >= deadline:
                    evt.set()
                    break
                time.sleep(0.1)

        budget_event = stop_event or threading.Event()
        stop_event = budget_event
        threading.Thread(
            target=_budget_guard,
            args=(budget_event, bootstrap_deadline),
            name="embedder-budget-guard",
            daemon=True,
        ).start()

    if bootstrap_mode and stop_event is not None and stop_event.is_set():
        reason = "bootstrap_budget_exhausted" if bootstrap_deadline is not None else "bootstrap_cancelled"
        return _record_bootstrap_placeholder(reason)

    placeholder = _guard_bootstrap_budget(
        budget_deadline=budget_deadline,
        stop_event=stop_event,
        bootstrap_mode=bootstrap_mode,
        requester=requester,
        stage="pre_lock",
    )
    if placeholder is not None:
        return placeholder

    lock = _embedder_lock()
    placeholder_embedder = None
    lock_timeout = LOCK_TIMEOUT
    if wait_override is not None:
        try:
            lock_timeout = min(lock_timeout, max(0.0, wait_override))
        except Exception:
            lock_timeout = LOCK_TIMEOUT

    if lock_timeout < 0:
        lock_timeout = 0.0

    # ``LOCK_TIMEOUT`` defaults to a fairly generous value because other parts
    # of the sandbox prefer long lived locks.  For the embedder initialisation
    # we cap the wait to the maximum embedder startup window so that stale lock
    # files left behind by crashed processes do not block the sandbox for
    # hours.  The cap mirrors the longest time the caller would otherwise wait
    # for the background initialisation thread, making sure both phases respect
    # the same upper bound.
    lock_cap = (
        _MAX_EMBEDDER_WAIT
        if max_wait_override is None
        else min(_MAX_EMBEDDER_WAIT, max_wait_override)
    )
    if _EMBEDDER_INIT_TIMEOUT >= 0:
        lock_cap = min(lock_cap, max(0.0, _EMBEDDER_INIT_TIMEOUT))
    if lock_cap >= 0 and lock_timeout > lock_cap:
        logger.warning(
            "capping embedder lock wait to %.0fs (requested %.0fs)",
            lock_cap,
            lock_timeout,
        )
        lock_timeout = lock_cap

    if lock is None:
        placeholder_embedder = _initialise_embedder_with_timeout(
            timeout_override=wait_override,
            suppress_timeout_log=wait_override is not None,
            requester=requester,
            stop_event=stop_event,
            fallback_on_timeout=bootstrap_mode or wait_override is not None,
            max_wait_override=max_wait_override,
            allow_background_wait=bootstrap_mode,
            load_budget_deadline=budget_deadline,
        )
        return _EMBEDDER if _EMBEDDER is not None else placeholder_embedder

    cleaned_once = False
    while True:
        try:
            with lock.acquire(timeout=lock_timeout):
                placeholder_embedder = _initialise_embedder_with_timeout(
                    timeout_override=wait_override,
                    suppress_timeout_log=wait_override is not None,
                    requester=requester,
                    stop_event=stop_event,
                    fallback_on_timeout=bootstrap_mode or wait_override is not None,
                    max_wait_override=max_wait_override,
                    allow_background_wait=bootstrap_mode,
                    load_budget_deadline=budget_deadline,
                )
                break
        except LockTimeout:
            _dump_embedder_thread("lock_timeout")
            lock_path = getattr(lock, "lock_file", "")
            if cleaned_once:
                logger.error(
                    "timed out waiting for embedder initialisation lock",
                    extra={"lock": lock_path, "requester": requester},
                )
                return None

            removed_lock = _cleanup_stale_embedder_lock(lock_path)
            removed_cache = _force_cleanup_embedder_locks()
            if not removed_lock and not removed_cache:
                logger.error(
                    "timed out waiting for embedder initialisation lock",
                    extra={"lock": lock_path, "requester": requester},
                )
                return None

            cleaned_once = True
            logger.warning(
                "retrying embedder initialisation after cleaning stale locks",
                extra={
                    "lock": lock_path,
                    "requester": requester,
                    "lock_removed": removed_lock,
                    "cache_locks_removed": removed_cache,
                },
            )
            _trace(
                "lock.retry",
                requester=requester,
                lock=lock_path,
                lock_removed=removed_lock,
                cache_removed=removed_cache,
            )
            continue
    return _EMBEDDER if _EMBEDDER is not None else placeholder_embedder


def governed_embed(
    text: str,
    embedder: SentenceTransformer | None = None,
    timeout: float | None = None,
) -> Optional[List[float]]:
    """Return an embedding vector for ``text`` with safety checks.

    The input text is first scanned for disallowed licences.  If any are
    detected the function returns ``None``.  Secrets are redacted before
    computing the embedding to avoid storing sensitive data in the vector
    space.  Any runtime failures during embedding are swallowed and ``None``
    is returned.
    """

    if not text:
        return None

    cleaned = redact(text)
    redacted = cleaned != text
    cleaned_empty = not cleaned.strip()

    lic = license_check(text)
    if lic:
        try:  # pragma: no cover - best effort logging
            logger.warning(
                "skipping embedding due to license %s", lic,
                extra={
                    "fingerprint": license_fingerprint(text),
                    "redacted": redacted,
                    "cleaned_empty": cleaned_empty,
                },
            )
        except Exception:
            logger.warning(
                "skipping embedding due to license %s (redacted=%s cleaned_empty=%s)",
                lic,
                redacted,
                cleaned_empty,
            )
        return None

    risks = find_semantic_risks(text.splitlines())
    if risks:
        logger.warning(
            "skipping embedding due to semantic risks: %s (redacted=%s cleaned_empty=%s)",
            [r[1] for r in risks],
            redacted,
            cleaned_empty,
        )
        return None
    if cleaned_empty:
        logger.warning(
            "skipping embedding because cleaned text is empty (redacted=%s cleaned_empty=%s)",
            redacted,
            cleaned_empty,
        )
        return None
    if redacted:
        logger.warning(
            "redacted secrets prior to embedding (redacted=%s cleaned_empty=%s)",
            redacted,
            cleaned_empty,
        )
    model = embedder or get_embedder(timeout=timeout)
    if model is None:
        return None

    def _encode_supports_truncation(model_obj: Any) -> tuple[bool, bool]:
        try:
            signature = inspect.signature(model_obj.encode)
        except (TypeError, ValueError):
            return False, False
        params = signature.parameters
        return "max_length" in params, "truncation" in params

    def _count_tokens(text: str, model_obj: Any) -> int:
        tokenizer = getattr(model_obj, "tokenizer", None)
        if tokenizer is not None:
            try:
                return len(tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                pass
        return max(1, len(text) // EMBEDDING_CHARS_PER_TOKEN)

    def _truncate_text_for_embedding(raw: str, model_obj: Any) -> str:
        max_tokens = 200
        tokenizer = getattr(model_obj, "tokenizer", None)
        if tokenizer is not None:
            try:
                token_ids = tokenizer.encode(raw, add_special_tokens=False)
                if len(token_ids) > max_tokens:
                    token_ids = token_ids[:max_tokens]
                    if hasattr(tokenizer, "decode"):
                        return tokenizer.decode(token_ids, skip_special_tokens=True)
            except Exception:
                pass
        words = raw.split()
        if len(words) > max_tokens:
            return " ".join(words[:max_tokens])
        return raw

    supports_max_length, supports_truncation = _encode_supports_truncation(model)
    encode_kwargs: dict[str, Any] = {}
    if supports_max_length:
        encode_kwargs["max_length"] = 200
    if supports_truncation:
        encode_kwargs["truncation"] = True
    cleaned_for_embedding = cleaned
    if len(cleaned_for_embedding) > EMBEDDING_CHAR_TRUNCATION_THRESHOLD:
        approx_tokens = max(1, len(cleaned_for_embedding) // EMBEDDING_CHARS_PER_TOKEN)
        logger.warning(
            "embedding input too long; truncating (chars=%s approx_tokens=%s threshold=%s)",
            len(cleaned_for_embedding),
            approx_tokens,
            EMBEDDING_CHAR_TRUNCATION_THRESHOLD,
        )
        cleaned_for_embedding = cleaned_for_embedding[:EMBEDDING_CHAR_TRUNCATION_THRESHOLD]
    hard_max_tokens = 200
    original_token_count = _count_tokens(cleaned_for_embedding, model)
    cleaned_for_embedding = _truncate_text_for_embedding(cleaned_for_embedding, model)
    truncated_token_count = _count_tokens(cleaned_for_embedding, model)
    if truncated_token_count > hard_max_tokens:
        logger.warning(
            "embedding input exceeds hard max tokens; returning zero vector "
            "(tokens=%s hard_max=%s truncation_supported=%s)",
            truncated_token_count,
            hard_max_tokens,
            supports_truncation,
        )
        return [0.0] * _STUB_EMBEDDER_DIMENSION
    if original_token_count > hard_max_tokens and not supports_truncation:
        logger.warning(
            "embedding input exceeds hard max tokens and truncation disabled; "
            "returning zero vector (tokens=%s hard_max=%s)",
            original_token_count,
            hard_max_tokens,
        )
        return [0.0] * _STUB_EMBEDDER_DIMENSION
    approx_tokens = max(1, len(cleaned_for_embedding) // EMBEDDING_CHARS_PER_TOKEN)
    logger.debug(
        "encoding embedding input (chars=%s approx_tokens=%s)",
        len(cleaned_for_embedding),
        approx_tokens,
    )
    try:  # pragma: no cover - external model may fail at runtime
        return model.encode([cleaned_for_embedding], **encode_kwargs)[0].tolist()
    except Exception:
        logger.exception(
            "embedding failed during model.encode (redacted=%s cleaned_empty=%s text_len=%s)",
            redacted,
            cleaned_empty,
            len(cleaned_for_embedding),
        )
        return [0.0] * _STUB_EMBEDDER_DIMENSION


def embedder_diagnostics() -> dict[str, Any]:
    """Return a snapshot of the embedder initialisation state."""

    thread = _EMBEDDER_INIT_THREAD
    diagnostics: dict[str, Any] = {
        "embedder_ready": _EMBEDDER is not None,
        "fallback_announced": _FALLBACK_ANNOUNCED,
        "embedder_disabled": _EMBEDDER_DISABLED,
        "timeout_logged": _EMBEDDER_TIMEOUT_LOGGED,
        "timeout_reached": _EMBEDDER_TIMEOUT_REACHED,
    }
    if _EMBEDDER is not None:
        diagnostics["embedder_type"] = type(_EMBEDDER).__name__
    if thread is not None:
        diagnostics["thread_alive"] = thread.is_alive()
        diagnostics["thread_name"] = thread.name
    else:
        diagnostics["thread_alive"] = False
    diagnostics["event_set"] = _EMBEDDER_INIT_EVENT.is_set()
    diagnostics["wait_cap"] = _MAX_EMBEDDER_WAIT
    diagnostics["init_timeout"] = _EMBEDDER_INIT_TIMEOUT
    return diagnostics


def cancel_embedder_initialisation(
    stop_event: threading.Event | None = None,
    *,
    reason: str = "requested",
    join_timeout: float = 2.0,
) -> None:
    """Signal the embedder initialisation thread to stop and wait briefly."""

    global _EMBEDDER_STOP_EVENT
    if stop_event is None:
        stop_event = _EMBEDDER_STOP_EVENT
    else:
        _EMBEDDER_STOP_EVENT = stop_event
    _cancel_embedder_initialisation(stop_event, reason=reason, join_timeout=join_timeout)


def rearm_bootstrap_embedder_placeholder() -> None:
    """Clear any bootstrap placeholder memo and allow initialisation attempts."""

    global _EMBEDDER_BOOTSTRAP_DEFERRED, _EMBEDDER_BOOTSTRAP_PLACEHOLDER

    _EMBEDDER_BOOTSTRAP_DEFERRED = False
    _EMBEDDER_BOOTSTRAP_PLACEHOLDER = None
    with _EMBEDDER_BOOTSTRAP_DEFERRALS_LOCK:
        _EMBEDDER_BOOTSTRAP_DEFERRALS.clear()


__all__ = [
    "DEFAULT_SENTENCE_TRANSFORMER_MODEL",
    "canonical_model_id",
    "SENTENCE_TRANSFORMER_DEVICE",
    "governed_embed",
    "get_embedder",
    "cancel_embedder_initialisation",
    "rearm_bootstrap_embedder_placeholder",
    "disable_embedder",
    "embedder_diagnostics",
    "initialise_sentence_transformer",
]
