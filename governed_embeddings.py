from __future__ import annotations

import errno
import contextlib
import inspect
import logging
import os
import shutil
import tarfile
import threading
import time
from pathlib import Path
from typing import Any, List, Optional, TYPE_CHECKING, cast, Set

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
_MODEL_NAME = "all-MiniLM-L6-v2"
_EMBEDDER_INIT_TIMEOUT = float(os.getenv("EMBEDDER_INIT_TIMEOUT", "180"))
_MAX_EMBEDDER_WAIT = float(os.getenv("EMBEDDER_INIT_MAX_WAIT", "180"))
_SOFT_EMBEDDER_WAIT = float(os.getenv("EMBEDDER_INIT_SOFT_WAIT", "30"))
_EMBEDDER_INIT_EVENT = threading.Event()
_EMBEDDER_INIT_THREAD: threading.Thread | None = None
_EMBEDDER_TIMEOUT_LOGGED = False
_EMBEDDER_WAIT_CAPPED = False
_EMBEDDER_SOFT_WAIT_LOGGED = False
_EMBEDDER_TIMEOUT_REACHED = False
_FALLBACK_ANNOUNCED = False
_HF_LOCK_CLEANUP_TIMEOUT = float(os.getenv("HF_LOCK_CLEANUP_TIMEOUT", "5"))
_BUNDLED_EMBEDDER: Any | None = None
_BUNDLED_EMBEDDER_LOCK = threading.Lock()
_EMBEDDER_REQUESTER_TIMEOUTS: Set[str] = set()
_EMBEDDER_REQUESTER_LOGGED: Set[str] = set()
_EMBEDDER_REQUESTER_LOCK = threading.Lock()
_STUB_EMBEDDER_DIMENSION = 384
_STUB_FALLBACK_USED = False


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
            with tarfile.open(archive) as tar:
                tar.extractall(tmp_dir)
            if target_dir.exists():
                shutil.rmtree(target_dir, ignore_errors=True)
            tmp_dir.rename(target_dir)
        except Exception as exc:
            logger.warning("failed to extract bundled embedder archive: %s", exc)
            with contextlib.suppress(Exception):
                shutil.rmtree(tmp_dir, ignore_errors=True)
            return None
    return target_dir


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
        return None

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


def _cleanup_hf_locks(cache_dir: Path, *, focus: Path | None = None) -> None:
    """Remove stale Hugging Face lock files left behind by crashed downloads.

    ``focus`` allows callers to restrict the scan to a specific subtree (for
    example the cache directory of the embedder being loaded).  When provided it
    takes precedence over the broader ``cache_dir`` walk which can be expensive
    in shared caches with many unrelated models.  The fallback to scanning the
    entire cache is preserved for backwards compatibility when ``focus`` is not
    supplied or does not exist.
    """

    if _HF_LOCK_CLEANUP_TIMEOUT == 0:
        return

    if not cache_dir.exists():
        return

    search_roots: list[Path] = []
    if focus is not None and focus.exists():
        search_roots.append(focus)
    else:
        search_roots.append(cache_dir)

    if cache_dir not in search_roots:
        search_roots.append(cache_dir)

    deadline: float | None = None
    if _HF_LOCK_CLEANUP_TIMEOUT > 0:
        deadline = time.monotonic() + _HF_LOCK_CLEANUP_TIMEOUT

    try:
        for base in search_roots:
            if not base.exists():
                continue

            for root, dirs, files in os.walk(base):
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
                            stale = time.time() - lock_path.stat().st_mtime > max(
                                LOCK_TIMEOUT, 300
                            )
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
                if deadline is not None and time.monotonic() >= deadline:
                    break
            if deadline is not None and time.monotonic() >= deadline:
                break
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


def _ensure_embedder_thread_locked() -> threading.Event:
    """Ensure the background embedder initialisation thread is running."""

    global _EMBEDDER_INIT_THREAD, _EMBEDDER_TIMEOUT_LOGGED, _EMBEDDER_TIMEOUT_REACHED

    if _EMBEDDER_INIT_THREAD is not None and _EMBEDDER_INIT_THREAD.is_alive():
        return _EMBEDDER_INIT_EVENT

    _EMBEDDER_INIT_EVENT.clear()
    _EMBEDDER_TIMEOUT_LOGGED = False
    _EMBEDDER_TIMEOUT_REACHED = False
    global _FALLBACK_ANNOUNCED
    _FALLBACK_ANNOUNCED = False

    def _initialise() -> None:
        global _EMBEDDER, _EMBEDDER_TIMEOUT_LOGGED
        start = time.perf_counter()
        logger.info("embedder initialisation thread started", extra={"model": _MODEL_NAME})
        try:
            result = _load_embedder()
            if result is not None:
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


def _initialise_embedder_with_timeout(
    timeout_override: float | None = None,
    *,
    suppress_timeout_log: bool = False,
    requester: str | None = None,
) -> None:
    """Initialise the shared embedder without blocking indefinitely.

    ``timeout_override`` allows callers to shorten the wait period without
    affecting the global timeout configuration.  When an override is used the
    function returns ``None`` once the shorter timeout elapses while keeping the
    background initialisation thread alive.
    """

    global _EMBEDDER_TIMEOUT_LOGGED, _EMBEDDER_SOFT_WAIT_LOGGED, _EMBEDDER_TIMEOUT_REACHED

    with _EMBEDDER_THREAD_LOCK:
        if _EMBEDDER is not None:
            return
        event = _ensure_embedder_thread_locked()

    if _EMBEDDER_TIMEOUT_REACHED and not event.is_set():
        if suppress_timeout_log:
            logger.debug(
                "skipping embedder wait after previous timeout",
                extra={"model": _MODEL_NAME},
            )
        else:
            logger.debug(
                "embedder initialisation previously timed out; not waiting",
                extra={"model": _MODEL_NAME},
            )
        return

    global _EMBEDDER_WAIT_CAPPED

    requested_timeout = _EMBEDDER_INIT_TIMEOUT
    if timeout_override is not None:
        requested_timeout = min(requested_timeout, max(0.0, timeout_override))

    wait_cap = max(0.0, min(requested_timeout, _MAX_EMBEDDER_WAIT))
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
                extra={"model": _MODEL_NAME},
            )
        elif not requester:
            logger.debug(
                "waiting up to %.1fs for embedder initialisation",
                wait_time,
                extra={"model": _MODEL_NAME},
            )
    finished = event.wait(wait_time)
    if finished:
        _EMBEDDER_TIMEOUT_LOGGED = False
        _EMBEDDER_TIMEOUT_REACHED = False
        if requester:
            with _EMBEDDER_REQUESTER_LOCK:
                _EMBEDDER_REQUESTER_TIMEOUTS.discard(requester)
        return

    if _EMBEDDER is None and _activate_bundled_fallback("timeout"):
        _EMBEDDER_TIMEOUT_LOGGED = False
        _EMBEDDER_TIMEOUT_REACHED = False
        if requester:
            with _EMBEDDER_REQUESTER_LOCK:
                _EMBEDDER_REQUESTER_TIMEOUTS.add(requester)
        return

    if suppress_timeout_log:
        if wait_time > 0:
            if requester:
                logger.debug(
                    "sentence transformer initialisation still pending after %.0fs (requested by %s)",
                    wait_time,
                    requester,
                    extra={"model": _MODEL_NAME},
                )
            else:
                logger.debug(
                    "sentence transformer initialisation still pending after %.0fs",
                    wait_time,
                    extra={"model": _MODEL_NAME},
                )
        _EMBEDDER_TIMEOUT_REACHED = True
        if requester:
            with _EMBEDDER_REQUESTER_LOCK:
                _EMBEDDER_REQUESTER_TIMEOUTS.add(requester)
        return

    if not _EMBEDDER_TIMEOUT_LOGGED:
        _EMBEDDER_TIMEOUT_LOGGED = True
        _EMBEDDER_TIMEOUT_REACHED = True
        if requester:
            logger.error(
                "sentence transformer initialisation exceeded %.0fs; continuing without embeddings (requested by %s)",
                _EMBEDDER_INIT_TIMEOUT,
                requester,
                extra={"model": _MODEL_NAME},
            )
        else:
            logger.error(
                "sentence transformer initialisation exceeded %.0fs; continuing without embeddings",
                _EMBEDDER_INIT_TIMEOUT,
                extra={"model": _MODEL_NAME},
            )
        if requester:
            with _EMBEDDER_REQUESTER_LOCK:
                _EMBEDDER_REQUESTER_TIMEOUTS.add(requester)


def _load_embedder() -> SentenceTransformer | None:
    """Load the shared ``SentenceTransformer`` instance with offline fallbacks."""

    if SentenceTransformer is None:  # pragma: no cover - optional dependency missing
        fallback = _load_bundled_embedder()
        if fallback is not None:
            return cast("SentenceTransformer", fallback)
        return None

    cache_dir = _cache_base()
    local_kwargs: dict[str, object] = {}
    start = time.perf_counter()
    logger.info(
        "starting sentence transformer initialisation",
        extra={
            "model": _MODEL_NAME,
            "cache_dir": str(cache_dir) if cache_dir is not None else None,
        },
    )
    if cache_dir is not None:
        model_cache = _cached_model_path(cache_dir, _MODEL_NAME)
        focus_dir = model_cache if model_cache.exists() else model_cache.parent
        try:
            _cleanup_hf_locks(cache_dir, focus=focus_dir)
        except TypeError:
            # Tests stub ``_cleanup_hf_locks`` with a positional-only lambda.
            # Fallback to the legacy calling convention to keep compatibility.
            _cleanup_hf_locks(cache_dir)
        local_kwargs["cache_folder"] = str(cache_dir)
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
                logger.info(
                    "loading sentence transformer from cached snapshot",
                    extra={"model": _MODEL_NAME, "snapshot": str(snapshot_path)},
                )
                model = SentenceTransformer(
                    str(snapshot_path), local_files_only=True
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
                return model
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
        logger.info(
            "loading sentence transformer via hub",
            extra={
                "model": _MODEL_NAME,
                "local_files_only": local_kwargs.get("local_files_only", False),
            },
        )
        model = SentenceTransformer(_MODEL_NAME, **local_kwargs)
        duration = time.perf_counter() - start
        logger.info(
            "loaded sentence transformer",
            extra={"model": _MODEL_NAME, "duration": round(duration, 3)},
        )
        return model
    except Exception as exc:
        if local_kwargs.pop("local_files_only", None):
            if os.environ.get("HF_HUB_OFFLINE") or os.environ.get("TRANSFORMERS_OFFLINE"):
                logger.warning(
                    "sentence transformer initialisation failed in offline mode: %s",
                    exc,
                )
                fallback = _load_bundled_embedder()
                if fallback is not None:
                    return cast("SentenceTransformer", fallback)
                return None
            try:
                logger.info(
                    "retrying sentence transformer load with hub access",
                    extra={"model": _MODEL_NAME},
                )
                model = SentenceTransformer(_MODEL_NAME, **local_kwargs)
                duration = time.perf_counter() - start
                logger.info(
                    "loaded sentence transformer after retry",
                    extra={"model": _MODEL_NAME, "duration": round(duration, 3)},
                )
                return model
            except Exception:
                logger.warning("failed to initialise sentence transformer: %s", exc)
                fallback = _load_bundled_embedder()
                if fallback is not None:
                    return cast("SentenceTransformer", fallback)
                return None
        logger.warning("failed to initialise sentence transformer: %s", exc)
        fallback = _load_bundled_embedder()
        if fallback is not None:
            return cast("SentenceTransformer", fallback)
        return None


def get_embedder(timeout: float | None = None) -> SentenceTransformer | None:
    """Return a lazily-instantiated shared :class:`SentenceTransformer`.

    The model is created on first use and reused for subsequent calls.  Errors
    during initialisation result in ``None`` being cached to avoid repeated
    expensive failures.  When ``timeout`` is provided the caller-specific wait
    duration is capped to that value, allowing services that run during startup
    to avoid blocking for the full global timeout.
    """
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER

    requester = _identify_embedder_requester()
    lock = _embedder_lock()
    wait_override = timeout
    lock_timeout = LOCK_TIMEOUT
    if wait_override is not None:
        try:
            lock_timeout = min(lock_timeout, max(0.0, wait_override))
        except Exception:
            lock_timeout = LOCK_TIMEOUT

    if lock is None:
        _initialise_embedder_with_timeout(
            timeout_override=wait_override,
            suppress_timeout_log=wait_override is not None,
            requester=requester,
        )
        return _EMBEDDER

    try:
        with lock.acquire(timeout=lock_timeout):
            _initialise_embedder_with_timeout(
                timeout_override=wait_override,
                suppress_timeout_log=wait_override is not None,
                requester=requester,
            )
    except LockTimeout:
        lock_path = getattr(lock, "lock_file", "")
        logger.error(
            "timed out waiting for embedder initialisation lock",
            extra={"lock": lock_path, "requester": requester},
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
