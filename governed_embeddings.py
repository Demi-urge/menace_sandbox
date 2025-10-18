from __future__ import annotations

import errno
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, List, Optional, TYPE_CHECKING

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

try:  # pragma: no cover - optional heavy dependency
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - simplify in environments without the package
    SentenceTransformer = None  # type: ignore

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
_MODEL_NAME = "all-MiniLM-L6-v2"
_EMBEDDER_INIT_TIMEOUT = float(os.getenv("EMBEDDER_INIT_TIMEOUT", "180"))
_MAX_EMBEDDER_WAIT = float(os.getenv("EMBEDDER_INIT_MAX_WAIT", "180"))
_SOFT_EMBEDDER_WAIT = float(os.getenv("EMBEDDER_INIT_SOFT_WAIT", "30"))
_EMBEDDER_INIT_EVENT = threading.Event()
_EMBEDDER_INIT_THREAD: threading.Thread | None = None
_EMBEDDER_TIMEOUT_LOGGED = False
_EMBEDDER_WAIT_CAPPED = False
_EMBEDDER_SOFT_WAIT_LOGGED = False
_HF_LOCK_CLEANUP_TIMEOUT = float(os.getenv("HF_LOCK_CLEANUP_TIMEOUT", "5"))


def _cache_base() -> Optional[Path]:
    """Return the configured Hugging Face cache directory when available."""

    for env in ("TRANSFORMERS_CACHE", "HF_HOME"):
        loc = os.getenv(env)
        if loc:
            return Path(loc).expanduser()
    default = Path.home() / ".cache" / "huggingface"
    return default if default.exists() else None


def _cached_model_path(cache_dir: Path, model_name: str) -> Path:
    """Return the expected cache path for ``model_name`` within ``cache_dir``."""

    safe_name = model_name.replace("/", "--")
    return cache_dir / "hub" / f"models--sentence-transformers--{safe_name}"


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

    for _, path in sorted(candidates, key=lambda item: item[0], reverse=True):
        config = path / "config.json"
        try:
            if config.exists():
                return path
        except FileNotFoundError:
            continue
        except Exception as exc:  # pragma: no cover - diagnostics only
            logger.debug("failed to inspect snapshot %s: %s", path, exc)
            continue
    return None


def _cleanup_hf_locks(cache_dir: Path) -> None:
    """Remove stale Hugging Face lock files left behind by crashed downloads."""

    if _HF_LOCK_CLEANUP_TIMEOUT == 0:
        return

    if not cache_dir.exists():
        return

    deadline: float | None = None
    if _HF_LOCK_CLEANUP_TIMEOUT > 0:
        deadline = time.monotonic() + _HF_LOCK_CLEANUP_TIMEOUT

    try:
        for root, dirs, files in os.walk(cache_dir):
            if deadline is not None and time.monotonic() >= deadline:
                dirs[:] = []
                logger.debug(
                    "aborting huggingface lock cleanup after %.1fs", _HF_LOCK_CLEANUP_TIMEOUT
                )
                break

            for name in files:
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

                stale = False
                if is_lock_stale is not None:
                    try:
                        stale = is_lock_stale(
                            str(lock_path), timeout=max(LOCK_TIMEOUT, 300)
                        )
                    except Exception as exc:  # pragma: no cover - diagnostics only
                        logger.debug("failed to check lock %s: %s", lock_path, exc)
                        stale = False
                if not stale:
                    try:
                        stale = time.time() - lock_path.stat().st_mtime > max(LOCK_TIMEOUT, 300)
                    except FileNotFoundError:
                        continue
                    except Exception as exc:  # pragma: no cover - diagnostics only
                        logger.debug("failed to stat lock file %s: %s", lock_path, exc)
                        continue

                if not stale:
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
    except Exception as exc:  # pragma: no cover - best effort logging
        logger.debug("failed to scan huggingface cache for locks: %s", exc)


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


def _ensure_embedder_thread_locked() -> threading.Event:
    """Ensure the background embedder initialisation thread is running."""

    global _EMBEDDER_INIT_THREAD

    if _EMBEDDER_INIT_THREAD is not None and _EMBEDDER_INIT_THREAD.is_alive():
        return _EMBEDDER_INIT_EVENT

    _EMBEDDER_INIT_EVENT.clear()

    def _initialise() -> None:
        global _EMBEDDER, _EMBEDDER_TIMEOUT_LOGGED
        try:
            _EMBEDDER = _load_embedder()
        except Exception:  # pragma: no cover - defensive logging
            logger.debug("sentence transformer initialisation raised", exc_info=True)
            _EMBEDDER = None
        finally:
            if _EMBEDDER is not None and _EMBEDDER_TIMEOUT_LOGGED:
                logger.info("sentence transformer became available after initial timeout")
            _EMBEDDER_INIT_EVENT.set()

    _EMBEDDER_INIT_THREAD = threading.Thread(
        target=_initialise,
        name="menace-embedder-init",
        daemon=True,
    )
    _EMBEDDER_INIT_THREAD.start()
    return _EMBEDDER_INIT_EVENT


def _initialise_embedder_with_timeout() -> None:
    """Initialise the shared embedder without blocking indefinitely."""

    global _EMBEDDER_TIMEOUT_LOGGED, _EMBEDDER_SOFT_WAIT_LOGGED

    with _EMBEDDER_THREAD_LOCK:
        if _EMBEDDER is not None:
            return
        event = _ensure_embedder_thread_locked()

    global _EMBEDDER_WAIT_CAPPED

    wait_cap = max(0.0, min(_EMBEDDER_INIT_TIMEOUT, _MAX_EMBEDDER_WAIT))
    wait_limit = wait_cap
    soft_clamped = False
    if _SOFT_EMBEDDER_WAIT >= 0:
        wait_limit = min(wait_limit, _SOFT_EMBEDDER_WAIT)
        soft_clamped = wait_limit < wait_cap

    if (
        not _EMBEDDER_WAIT_CAPPED
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

    wait_time = wait_limit if not _EMBEDDER_TIMEOUT_LOGGED else 0.0
    finished = event.wait(wait_time)
    if finished:
        _EMBEDDER_TIMEOUT_LOGGED = False
        return

    if not _EMBEDDER_TIMEOUT_LOGGED:
        _EMBEDDER_TIMEOUT_LOGGED = True
        logger.error(
            "sentence transformer initialisation exceeded %.0fs; continuing without embeddings",
            _EMBEDDER_INIT_TIMEOUT,
        )


def _load_embedder() -> SentenceTransformer | None:
    """Load the shared ``SentenceTransformer`` instance with offline fallbacks."""

    if SentenceTransformer is None:  # pragma: no cover - optional dependency missing
        return None

    cache_dir = _cache_base()
    local_kwargs: dict[str, object] = {}
    if cache_dir is not None:
        _cleanup_hf_locks(cache_dir)
        local_kwargs["cache_folder"] = str(cache_dir)
        model_cache = _cached_model_path(cache_dir, _MODEL_NAME)
        snapshot_path = _resolve_local_snapshot(model_cache) if model_cache.exists() else None
        offline_env = os.environ.get("HF_HUB_OFFLINE") or os.environ.get("TRANSFORMERS_OFFLINE")
        force_local = os.environ.get("SANDBOX_FORCE_LOCAL_EMBEDDER", "").lower() not in {
            "",
            "0",
            "false",
        }
        if offline_env or force_local or snapshot_path is not None:
            local_kwargs["local_files_only"] = True

        if snapshot_path is not None:
            try:
                return SentenceTransformer(
                    str(snapshot_path), local_files_only=True
                )
            except Exception as exc:  # pragma: no cover - diagnostics only
                logger.warning(
                    "failed to load sentence transformer from cached snapshot %s: %s",
                    snapshot_path,
                    exc,
                )

    token = os.getenv("HUGGINGFACE_API_TOKEN")
    if token:
        # The transformers stack honours these environment variables directly and
        # avoids the interactive ``huggingface_hub.login`` flow that can hang in
        # restricted environments.
        os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", token)
        os.environ.setdefault("HF_HUB_TOKEN", token)

    try:
        return SentenceTransformer(_MODEL_NAME, **local_kwargs)
    except Exception as exc:
        if local_kwargs.pop("local_files_only", None):
            if os.environ.get("HF_HUB_OFFLINE") or os.environ.get("TRANSFORMERS_OFFLINE"):
                logger.warning(
                    "sentence transformer initialisation failed in offline mode: %s",
                    exc,
                )
                return None
            try:
                return SentenceTransformer(_MODEL_NAME, **local_kwargs)
            except Exception:
                logger.warning("failed to initialise sentence transformer: %s", exc)
                return None
        logger.warning("failed to initialise sentence transformer: %s", exc)
        return None


def get_embedder() -> SentenceTransformer | None:
    """Return a lazily-instantiated shared :class:`SentenceTransformer`.

    The model is created on first use and reused for subsequent calls.  Errors
    during initialisation result in ``None`` being cached to avoid repeated
    expensive failures.
    """
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER

    lock = _embedder_lock()
    timeout = LOCK_TIMEOUT

    if lock is None:
        _initialise_embedder_with_timeout()
        return _EMBEDDER

    try:
        with lock.acquire(timeout=timeout):
            _initialise_embedder_with_timeout()
    except LockTimeout:
        lock_path = getattr(lock, "lock_file", "")
        logger.error(
            "timed out waiting for embedder initialisation lock", extra={"lock": lock_path}
        )
        return None
    return _EMBEDDER


def governed_embed(text: str, embedder: SentenceTransformer | None = None) -> Optional[List[float]]:
    """Return an embedding vector for ``text`` with safety checks.

    The input text is first scanned for disallowed licences.  If any are
    detected the function returns ``None``.  Secrets are redacted before
    computing the embedding to avoid storing sensitive data in the vector
    space.  Any runtime failures during embedding are swallowed and ``None``
    is returned.
    """

    if not text:
        return None
    lic = license_check(text)
    if lic:
        try:  # pragma: no cover - best effort logging
            logger.warning(
                "skipping embedding due to license %s", lic,
                extra={"fingerprint": license_fingerprint(text)},
            )
        except Exception:
            logger.warning("skipping embedding due to license %s", lic)
        return None

    risks = find_semantic_risks(text.splitlines())
    if risks:
        logger.warning("skipping embedding due to semantic risks: %s", [r[1] for r in risks])
        return None
    cleaned = redact(text)
    if cleaned != text:
        logger.warning("redacted secrets prior to embedding")
    model = embedder or get_embedder()
    if model is None:
        return None
    try:  # pragma: no cover - external model may fail at runtime
        return model.encode([cleaned])[0].tolist()
    except Exception:
        return None


__all__ = ["governed_embed", "get_embedder"]
