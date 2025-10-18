from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import suppress
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
_EMBEDDER_INIT_EVENT = threading.Event()
_EMBEDDER_INIT_THREAD: threading.Thread | None = None
_EMBEDDER_TIMEOUT_LOGGED = False


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


def _cleanup_hf_locks(cache_dir: Path) -> None:
    """Remove stale Hugging Face lock files left behind by crashed downloads."""

    if not cache_dir.exists():
        return

    try:
        lock_files = list(cache_dir.rglob("*.lock"))
    except Exception as exc:  # pragma: no cover - best effort logging
        logger.debug("failed to scan huggingface cache for locks: %s", exc)
        return

    now = time.time()
    for lock_path in lock_files:
        if lock_path.name == "menace-embedder.lock":
            continue
        try:
            if lock_path.is_dir():
                continue
        except Exception:  # pragma: no cover - ignore racing filesystem errors
            continue

        stale = False
        if is_lock_stale is not None:
            with suppress(Exception):
                if is_lock_stale(str(lock_path), timeout=LOCK_TIMEOUT):
                    stale = True
        if not stale:
            try:
                stale = now - lock_path.stat().st_mtime > max(LOCK_TIMEOUT, 300)
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
            logger.debug("failed to remove stale huggingface lock %s: %s", lock_path, exc)


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

    global _EMBEDDER_TIMEOUT_LOGGED

    with _EMBEDDER_THREAD_LOCK:
        if _EMBEDDER is not None:
            return
        event = _ensure_embedder_thread_locked()

    wait_time = _EMBEDDER_INIT_TIMEOUT if not _EMBEDDER_TIMEOUT_LOGGED else 0.0
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
        if os.environ.get("HF_HUB_OFFLINE") or os.environ.get("TRANSFORMERS_OFFLINE"):
            local_kwargs["local_files_only"] = True
        elif model_cache.exists():
            # We already have the files locally; avoid slow network calls.
            local_kwargs["local_files_only"] = True

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
