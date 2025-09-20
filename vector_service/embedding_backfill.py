from __future__ import annotations

"""Utilities for backfilling vector embeddings across databases."""

import logging
import time
from typing import Iterable, List, Sequence, Callable, Any, Dict
import importlib
import asyncio
from pathlib import Path
import json
import sys
import hashlib
import queue
import threading
import contextlib
from datetime import datetime, timedelta

from . import registry as _registry

try:  # pragma: no cover - optional heavy dependency
    from .vectorizer import SharedVectorService
except Exception:  # pragma: no cover - lightweight fallback
    class SharedVectorService:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

        def vectorise_and_store(self, *args, **kwargs):
            pass

from .decorators import log_and_measure
from compliance.license_fingerprint import (
    check as license_check,
    fingerprint as license_fingerprint,
)

from dynamic_path_router import resolve_path


def _log_violation(path: str, lic: str, hash_: str) -> None:
    try:  # pragma: no cover - best effort
        CodeDB = importlib.import_module("code_database").CodeDB
        CodeDB().log_license_violation(path, lic, hash_)
    except Exception:
        pass


try:  # pragma: no cover - optional dependency for metrics
    from . import metrics_exporter as _me  # type: ignore
except Exception:  # pragma: no cover - fallback when running standalone
    import metrics_exporter as _me  # type: ignore

try:  # pragma: no cover - optional dependency for event handling
    from unified_event_bus import UnifiedEventBus  # type: ignore
except Exception:  # pragma: no cover - fallback when bus unavailable
    UnifiedEventBus = None  # type: ignore

_RUN_OUTCOME = _me.Gauge(
    "embedding_backfill_runs_total",
    "Outcomes of EmbeddingBackfill.run calls",
    labelnames=["status", "trigger"],
)
_RUN_DURATION = _me.Gauge(
    "embedding_backfill_run_duration_seconds",
    "Duration of EmbeddingBackfill.run calls",
    labelnames=["trigger"],
)
_RUN_SKIPPED = _me.Gauge(
    "embedding_backfill_skipped_total",
    "Records skipped during EmbeddingBackfill due to licensing",
    labelnames=["db", "license"],
)

_PROCESSED_RECORDS = _me.Gauge(
    "embedding_watcher_processed_total",
    "Records processed by embedding watchers",
    labelnames=["watcher"],
)
_FAILED_EMBEDDINGS = _me.Gauge(
    "embedding_watcher_failed_total",
    "Failed embedding attempts by watchers",
    labelnames=["watcher"],
)
_RUNTIME_ERRORS = _me.Gauge(
    "embedding_watcher_errors_total",
    "Runtime errors encountered by watchers",
    labelnames=["watcher"],
)

try:  # pragma: no cover - optional dependency
    from embeddable_db_mixin import EmbeddableDBMixin  # type: ignore
except Exception:  # pragma: no cover
    EmbeddableDBMixin = object  # type: ignore

# Registry describing databases capable of embedding backfills. The file
# ``embedding_registry.json`` lives alongside this module and maps a short name
# to a ``module`` and ``class`` implementing :class:`EmbeddableDBMixin`.
DEFAULT_REGISTRY = resolve_path("vector_service/embedding_registry.json")
_REGISTRY_FILE = DEFAULT_REGISTRY

# Persistent record of last successful vectorisation per database.  Historically
# this lived in the project root as ``embedding_timestamps.json`` but fresh
# sandboxes may not ship with the file.  ``resolve_path`` raises a
# ``FileNotFoundError`` in that scenario, which would cause the import of this
# module to fail and upstream callers to fall back to lightweight stubs.  To keep
# the vector service functional out of the box we create the timestamp file on
# demand next to the registry if it is missing.
try:
    _TIMESTAMP_FILE = resolve_path("embedding_timestamps.json")
except FileNotFoundError:
    _TIMESTAMP_FILE = DEFAULT_REGISTRY.with_name("embedding_timestamps.json")
    if not _TIMESTAMP_FILE.exists():  # pragma: no cover - best effort initialisation
        try:
            _TIMESTAMP_FILE.write_text("{}")
        except Exception:
            pass
_DB_FILE_MAP: Dict[str, str] = {}


def _load_timestamps() -> dict[str, float]:
    if _TIMESTAMP_FILE.exists():
        try:  # pragma: no cover - best effort
            return json.loads(_TIMESTAMP_FILE.read_text())
        except Exception:
            return {}
    return {}


def _store_timestamps(data: dict[str, float]) -> None:
    try:  # pragma: no cover - best effort
        _TIMESTAMP_FILE.write_text(json.dumps(data))
    except Exception:
        pass


# Minimum set of database kinds expected to support embeddings. These are
# used by :func:`_verify_registry` and other modules to recognise valid
# backfill targets. New kinds such as ``failure`` and ``research`` are
# explicitly listed here so historical records can be backfilled.
KNOWN_DB_KINDS = {
    "bot",
    "workflow",
    "enhancement",
    "error",
    "information",
    "code",
    "discrepancy",
    "failure",
    "research",
    "resource",
}

# Staleness detection thresholds
_STALE_RECORD_THRESHOLD = 10
_STALE_AGE = timedelta(days=30)


def _load_registry(path: Path | None = None) -> dict[str, tuple[str, str]]:
    """Return mapping of source name to ``(module, class)`` tuples.

    The mapping starts with entries discovered via
    :mod:`vector_service.registry`, which now scans for ``*Vectorizer`` classes
    automatically.  This means new modalities become available for backfills by
    simply adding a new vectoriser module; no source code changes are required.
    ``path`` is retained for backwards compatibility with the original JSON
    based registry and, when provided, entries from that file are merged over
    the dynamic registrations.
    """

    mapping = dict(_registry.get_db_registry())

    reg_path = path or _REGISTRY_FILE
    if reg_path.exists():  # merge any legacy JSON entries
        try:  # pragma: no cover - best effort
            data = json.loads(reg_path.read_text())
        except Exception:
            data = {}
        for key, value in data.items():
            mod = value.get("module")
            cls = value.get("class")
            if isinstance(mod, str) and isinstance(cls, str):
                mapping.setdefault(key, (mod, cls))
    return mapping


class EmbeddingBackfill:
    """Trigger embedding backfills on all known database classes."""

    def __init__(self, batch_size: int = 100, backend: str = "annoy") -> None:
        self.batch_size = batch_size
        self.backend = backend

    # ------------------------------------------------------------------
    def _load_known_dbs(self, names: List[str] | None = None) -> List[type]:
        """Load registered ``EmbeddableDBMixin`` implementations.

        The registry maps a short database name to the module and class providing
        the implementation.  Modules are imported dynamically so new databases
        can be added by editing the registry file.  When ``names`` is supplied
        the returned classes are filtered to those whose class name matches any
        entry.  Matching is case-insensitive and ignores plural forms or a
        trailing ``DB`` suffix.
        """

        subclasses: List[type] = []

        registry = _load_registry()
        for mod_name, cls_name in registry.values():
            try:
                mod = importlib.import_module(mod_name)
                cls = getattr(mod, cls_name)
            except Exception:
                continue
            if not issubclass(cls, EmbeddableDBMixin) or not hasattr(
                cls, "backfill_embeddings"
            ):
                continue
            subclasses.append(cls)

        if names:
            keys = [n.lower().rstrip("s") for n in names]
            filtered: List[type] = []
            for cls in subclasses:
                name = cls.__name__.lower()
                base = name[:-2] if name.endswith("db") else name
                for key in keys:
                    if name.startswith(key) or base.startswith(key):
                        filtered.append(cls)
                        break
            subclasses = filtered

        for cls in subclasses:
            for meth in ("iter_records", "vector"):
                if not callable(getattr(cls, meth, None)):
                    raise TypeError(f"{cls.__name__} missing required method {meth}")
        return subclasses

    # ------------------------------------------------------------------
    def _verify_registry(self, names: List[str] | None = None) -> None:
        """Ensure registry entries expose the expected EmbeddableDB interface."""

        registry = _load_registry()
        problems: list[str] = []
        pkg_root_path = resolve_path("vector_service")
        pkg_root = pkg_root_path.name
        parent = pkg_root_path.parent
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        to_check = set(names or registry.keys())
        for name in to_check:
            mod_cls = registry.get(name)
            if not mod_cls:
                if names is None and _REGISTRY_FILE == DEFAULT_REGISTRY:
                    problems.append(f"{name}: not registered")
                continue
            mod_name, cls_name = mod_cls
            try:
                mod = importlib.import_module(mod_name)
            except BaseException:
                try:
                    mod = importlib.import_module(f"{pkg_root}.{mod_name}")
                except BaseException as exc:  # pragma: no cover - defensive
                    if names is not None:
                        problems.append(f"{name}: import failed ({exc})")
                    continue
            try:
                cls = getattr(mod, cls_name)
            except BaseException as exc:  # pragma: no cover - defensive
                if names is not None:
                    problems.append(f"{name}: import failed ({exc})")
                continue
            if not issubclass(cls, EmbeddableDBMixin):
                problems.append(f"{name}: not EmbeddableDBMixin")
                continue
            if not callable(getattr(cls, "backfill_embeddings", None)):
                logging.getLogger(__name__).warning(
                    "%s: missing backfill_embeddings", name
                )
                continue
            missing = [m for m in ("iter_records", "vector") if not callable(getattr(cls, m, None))]
            if missing:
                problems.append(f"{name}: missing {', '.join(missing)}")
        if names is None and _REGISTRY_FILE == DEFAULT_REGISTRY:
            for name in sorted(KNOWN_DB_KINDS - registry.keys()):
                problems.append(f"{name}: not registered")
        if problems:
            raise TypeError("invalid EmbeddableDB registrations: " + "; ".join(problems))

    # ------------------------------------------------------------------
    @log_and_measure
    def _process_db(
        self,
        db: EmbeddableDBMixin,
        *,
        batch_size: int,
        session_id: str = "",
    ) -> List[tuple[str, str]]:
        if not hasattr(db, "iter_records") or not hasattr(db, "needs_refresh"):
            try:
                db.backfill_embeddings(batch_size=batch_size)  # type: ignore[call-arg]
            except Exception:  # pragma: no cover - defensive fallback
                try:
                    db.backfill_embeddings()  # type: ignore[call-arg]
                except Exception:
                    pass
            return []

        processed = 0
        skipped: List[tuple[str, str]] = []
        for record_id, record, kind in db.iter_records():
            if processed >= batch_size:
                break
            if not db.needs_refresh(record_id, record):
                continue
            text = record if isinstance(record, str) else str(record)
            lic = license_check(text)
            if lic:
                _log_violation(
                    str(record_id), lic, license_fingerprint(text)
                )
                _RUN_SKIPPED.labels(db.__class__.__name__, lic).inc()
                skipped.append((str(record_id), lic))
                continue
            try:
                db.add_embedding(record_id, record, kind)
            except Exception:  # pragma: no cover - best effort
                continue
            processed += 1
        return skipped

    def check_out_of_sync(
        self,
        *,
        dbs: List[str] | None = None,
        batch_size: int | None = None,
        backend: str | None = None,
    ) -> List[str]:
        """Trigger backfills for databases with stale embeddings.

        The method scans registered :class:`EmbeddableDBMixin` implementations
        and invokes :meth:`run` for any whose records require re-embedding.  A
        list of database class names that were found to be out of sync is
        returned so callers can log or otherwise act upon it.
        """

        be = backend or self.backend
        names = dbs
        out_of_sync: List[str] = []
        subclasses = self._load_known_dbs(names=names)
        for cls in subclasses:
            try:
                db = cls(vector_backend=be)  # type: ignore[call-arg]
            except Exception:
                try:
                    db = cls()  # type: ignore[call-arg]
                except Exception:
                    continue
            for record_id, record, _ in db.iter_records():
                if db.needs_refresh(record_id, record):
                    out_of_sync.append(cls.__name__)
                    break
        if out_of_sync:
            self.run(
                dbs=out_of_sync,
                batch_size=batch_size,
                backend=be,
                trigger="auto",
            )
        return out_of_sync

    # ------------------------------------------------------------------
    @log_and_measure
    def run(
        self,
        *,
        session_id: str = "",
        batch_size: int | None = None,
        backend: str | None = None,
        db: str | None = None,
        dbs: List[str] | None = None,
        trigger: str = "manual",
    ) -> None:
        """Backfill embeddings for ``EmbeddableDBMixin`` subclasses.

        If ``db`` or ``dbs`` is provided, only classes whose name matches any
        of the supplied values are processed. Matching is case-insensitive and
        ignores plural forms or a trailing ``DB`` suffix.
        """
        start = time.time()
        status = "success"
        try:
            bs = batch_size if batch_size is not None else self.batch_size
            be = backend or self.backend
            names = dbs or ([db] if db else None)
            self._verify_registry(names)
            subclasses = self._load_known_dbs(names=names)
            logger = logging.getLogger(__name__)
            total = len(subclasses)
            for idx, cls in enumerate(subclasses, 1):
                try:
                    db = cls(vector_backend=be)  # type: ignore[call-arg]
                except Exception:  # pragma: no cover - fallback
                    try:
                        db = cls()  # type: ignore[call-arg]
                    except Exception:
                        continue
                logger.info(
                    "Backfilling %s (%d/%d)",
                    cls.__name__,
                    idx,
                    total,
                    extra={"session_id": session_id},
                )
                try:
                    skipped = self._process_db(db, batch_size=bs, session_id=session_id)
                    if skipped:
                        for rid, lic in skipped:
                            logger.warning(
                                "skipped %s due to license %s",
                                rid,
                                lic,
                                extra={"session_id": session_id},
                            )
                except Exception:  # pragma: no cover - best effort
                    continue
        except Exception:
            status = "failure"
            _RUN_OUTCOME.labels(status, trigger).inc()
            _RUN_DURATION.labels(trigger).set(time.time() - start)
            raise
        _RUN_OUTCOME.labels(status, trigger).inc()
        _RUN_DURATION.labels(trigger).set(time.time() - start)

    # ------------------------------------------------------------------
    def watch(
        self,
        *,
        interval: float = 60.0,
        dbs: List[str] | None = None,
    ) -> contextlib.AbstractContextManager[None]:
        """Continuously monitor databases for new or modified records."""

        return watch_databases(interval=interval, dbs=dbs, backend=self.backend)

    def watch_events(
        self,
        *,
        bus: "UnifiedEventBus" | None = None,
        batch_size: int | None = None,
    ) -> contextlib.AbstractContextManager[None]:
        """Listen for database change events and trigger incremental backfills."""

        return watch_event_bus(
            bus=bus,
            backend=self.backend,
            batch_size=batch_size if batch_size is not None else 1,
        )


def check_staleness(dbs: List[str]) -> List[str]:
    """Check embedding metadata for staleness and backfill when necessary."""

    backfill = EmbeddingBackfill()
    subclasses = backfill._load_known_dbs(names=dbs)
    stale: List[str] = []
    cutoff = datetime.utcnow() - _STALE_AGE
    for cls in subclasses:
        try:
            db = cls(vector_backend=backfill.backend)  # type: ignore[call-arg]
        except Exception:
            try:
                db = cls()  # type: ignore[call-arg]
            except Exception:
                continue
        count = 0
        for meta in getattr(db, "_metadata", {}).values():
            version = meta.get("embedding_version")
            created = meta.get("created_at")
            try:
                outdated_version = int(version) != int(db.embedding_version)
            except Exception:
                outdated_version = True
            try:
                created_time = datetime.fromisoformat(str(created)) if created else None
            except Exception:
                created_time = None
            old = created_time is None or created_time < cutoff
            if outdated_version or old:
                count += 1
                if count > _STALE_RECORD_THRESHOLD:
                    stale.append(cls.__name__)
                    break
    if stale:
        backfill.run(dbs=stale, trigger="stale")
    return stale


async def consume_record_changes(
    *,
    bus: "UnifiedEventBus",
    backend: str,
    batch_size: int,
    stop_event: asyncio.Event | None = None,
) -> None:
    """Consume ``db.record_changed`` events and trigger backfills.

    Events are batched by database name to avoid redundant backfill runs when
    multiple records change in quick succession. Processing continues until
    ``stop_event`` is set, allowing callers to cancel the consumer.
    """

    backfill = EmbeddingBackfill(batch_size=batch_size, backend=backend)
    q: asyncio.Queue[str] = asyncio.Queue()
    pending: set[str] = set()

    async def _handle(_topic: str, event: object) -> None:
        kind = None
        if isinstance(event, dict):
            kind = event.get("db") or event.get("kind")
        elif isinstance(event, str):
            kind = event
        if kind:
            await q.put(str(kind))

    bus.subscribe_async("db.record_changed", _handle)
    bus.subscribe_async("embedding:backfill", _handle)

    logger = logging.getLogger(__name__)
    while stop_event is None or not stop_event.is_set():
        try:
            name = await asyncio.wait_for(q.get(), timeout=0.5)
        except asyncio.TimeoutError:
            continue
        pending.add(name)
        try:
            while True:
                pending.add(q.get_nowait())
        except asyncio.QueueEmpty:
            pass
        names = list(pending)
        pending.clear()
        try:
            await asyncio.to_thread(
                backfill.run,
                dbs=names,
                batch_size=batch_size,
                backend=backend,
                trigger="event",
            )
            _PROCESSED_RECORDS.labels("event_bus").inc()
        except Exception:  # pragma: no cover - best effort
            logger.exception("event-triggered backfill failed for %s", names)
            _RUNTIME_ERRORS.labels("event_bus").inc()


def _run_event_bus_watcher(
    *,
    bus: "UnifiedEventBus",
    backend: str,
    batch_size: int,
    stop: threading.Event,
) -> None:
    """Run async ``consume_record_changes`` within bus event loop."""

    loop = bus._loop
    asyncio.set_event_loop(loop)
    stop_evt = asyncio.Event()

    def _stopper() -> None:
        stop.wait()
        loop.call_soon_threadsafe(stop_evt.set)

    threading.Thread(target=_stopper, daemon=True).start()

    loop.run_until_complete(
        consume_record_changes(
            bus=bus, backend=backend, batch_size=batch_size, stop_event=stop_evt
        )
    )


def _run_databases_watcher(
    *,
    interval: float,
    dbs: List[str] | None,
    backend: str,
    stop: threading.Event,
) -> None:
    svc = SharedVectorService()
    seen: dict[str, dict[str, str]] = {}
    backfill = EmbeddingBackfill(backend=backend)
    q: queue.Queue[str] = queue.Queue()
    pending: set[str] = set()

    def _enqueue(kind: str) -> None:
        if kind not in pending:
            pending.add(kind)
            q.put(kind)

    event_thread: threading.Thread | None = None
    if UnifiedEventBus is not None:
        bus = UnifiedEventBus()

        def _handle(_topic: str, event: object) -> None:
            kind = None
            if isinstance(event, dict):
                kind = event.get("db") or event.get("kind")
            elif isinstance(event, str):
                kind = event
            if not kind:
                return
            name = str(kind).lower().rstrip("s")
            if name in {"code", "bot", "error", "workflow"}:
                _enqueue(str(kind))

        for topic in ("db:record_added", "db:record_updated"):
            bus.subscribe(topic, _handle)

        def _event_worker() -> None:
            logger = logging.getLogger(__name__)
            while not stop.is_set():
                try:
                    kind = q.get(timeout=0.5)
                except queue.Empty:
                    continue
                try:
                    backfill.run(
                        dbs=[kind],
                        batch_size=1,
                        backend=backend,
                        trigger="event",
                    )
                    _PROCESSED_RECORDS.labels("event_bus").inc()
                except Exception:  # pragma: no cover - best effort
                    logger.exception(
                        "event-triggered backfill failed for %s", kind
                    )
                    _RUNTIME_ERRORS.labels("event_bus").inc()
                finally:
                    pending.discard(kind)

        event_thread = threading.Thread(target=_event_worker, daemon=True)
        event_thread.start()

    logger = logging.getLogger(__name__)
    try:
        while not stop.is_set():
            check_staleness(dbs or [])
            subclasses = backfill._load_known_dbs(names=dbs)
            for cls in subclasses:
                try:
                    db = cls(vector_backend=backend)  # type: ignore[call-arg]
                except Exception:
                    try:
                        db = cls()  # type: ignore[call-arg]
                    except Exception:  # pragma: no cover - best effort
                        continue
                key = cls.__name__
                cache = seen.setdefault(key, {})
                for record_id, record, kind in getattr(db, "iter_records", lambda: [])():
                    rid = str(record_id)
                    try:
                        raw = json.dumps(record, sort_keys=True, default=str)
                    except Exception:
                        raw = str(record)
                    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
                    if cache.get(rid) == digest:
                        continue
                    try:
                        svc.vectorise_and_store(kind, rid, record)
                        cache[rid] = digest
                        _PROCESSED_RECORDS.labels("db").inc()
                    except Exception:  # pragma: no cover - best effort
                        logger.exception(
                            "failed to vectorise record %s from %s", rid, key
                        )
                        _FAILED_EMBEDDINGS.labels("db").inc()
            stop.wait(interval)
    except Exception:  # pragma: no cover - best effort
        logger.exception("database watcher crashed")
        _RUNTIME_ERRORS.labels("db").inc()
    finally:
        if event_thread is not None:
            event_thread.join()


@contextlib.contextmanager
def watch_event_bus(
    *,
    bus: "UnifiedEventBus" | None = None,
    backend: str = "annoy",
    batch_size: int = 1,
):
    """Listen for database change events via :class:`UnifiedEventBus`."""

    if UnifiedEventBus is None:
        raise RuntimeError("UnifiedEventBus unavailable")

    bus = bus or UnifiedEventBus()
    stop = threading.Event()
    t = threading.Thread(
        target=_run_event_bus_watcher,
        kwargs={"bus": bus, "backend": backend, "batch_size": batch_size, "stop": stop},
        daemon=True,
    )
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join()


@contextlib.contextmanager
def watch_databases(
    *,
    interval: float = 60.0,
    dbs: List[str] | None = None,
    backend: str = "annoy",
):
    """Continuously monitor databases for new or modified records."""

    stop = threading.Event()
    t = threading.Thread(
        target=_run_databases_watcher,
        kwargs={"interval": interval, "dbs": dbs, "backend": backend, "stop": stop},
        daemon=True,
    )
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join()


async def schedule_backfill(
    *,
    batch_size: int | None = None,
    backend: str | None = None,
    dbs: Sequence[str] | None = None,
) -> None:
    """Asynchronously run :meth:`EmbeddingBackfill.run` for known databases.

    A single :class:`EmbeddingBackfill` instance is created and its
    :meth:`run` method is executed concurrently for each discovered
    :class:`EmbeddableDBMixin` subclass.  ``dbs`` can restrict execution to a
    subset of database names.
    """

    backfill = EmbeddingBackfill()
    if batch_size is not None:
        backfill.batch_size = batch_size
    if backend is not None:
        backfill.backend = backend

    names = list(dbs) if dbs else list(_registry.get_db_registry().keys())

    async def _run(name: str) -> None:
        await asyncio.to_thread(backfill.run, db=name)

    await asyncio.gather(*[_run(name) for name in names])


class StaleEmbeddingsError(RuntimeError):
    """Raised when embeddings remain stale after refresh attempts.

    Parameters
    ----------
    stale_dbs:
        Mapping of database name to the reason it is considered stale.
    """

    def __init__(self, stale_dbs: dict[str, str]):
        self.stale_dbs = stale_dbs
        detail = ", ".join(f"{n} ({r})" for n, r in stale_dbs.items())
        super().__init__(f"embeddings missing for: {detail}")


def ensure_embeddings_fresh(
    dbs: Iterable[str],
    *,
    retries: int = 2,
    delay: float = 0.5,
    return_details: bool = False,
    log_hook: Callable[[Dict[str, Dict[str, Any]]], None] | None = None,
) -> Dict[str, Dict[str, Any]] | None:
    """Ensure embedding metadata is present and up to date for ``dbs``.

    Parameters
    ----------
    dbs:
        Iterable of database names to check.
    retries:
        Number of times to attempt backfilling before raising an error.
    delay:
        Delay between backfill attempts.
    return_details:
        When ``True``, a mapping of databases requiring backfills and their
        diagnostics is returned.
    log_hook:
        Optional callable invoked with the diagnostics mapping before any
        backfills are scheduled. This enables external monitoring tools to
        observe embedding freshness checks.
    """

    names = [d for d in dbs if d]
    if not names:
        return

    logger = logging.getLogger(__name__)
    timestamps = _load_timestamps()

    def _needs_backfill(check: Iterable[str]) -> Dict[str, Dict[str, Any]]:
        registry = _load_registry()
        pending: Dict[str, Dict[str, Any]] = {}
        for name in check:
            cls = None
            mod_cls = registry.get(name)
            db_file = f"{name}.db"
            if mod_cls:
                mod_name, cls_name = mod_cls
                try:
                    mod = importlib.import_module(mod_name)
                    cls = getattr(mod, cls_name)
                    db_file = getattr(cls, "DB_FILE", getattr(cls, "DB_PATH", db_file))
                except Exception:
                    cls = None

            db_path = resolve_path(db_file)
            meta_path = resolve_path(f"{name}_embeddings.json")
            try:
                db_mtime = db_path.stat().st_mtime
            except FileNotFoundError:
                continue

            last_vec = float(timestamps.get(name, 0.0))
            info: Dict[str, Any] = {
                "db_mtime": db_mtime,
                "last_vectorization": last_vec,
            }
            if last_vec < db_mtime:
                info["reason"] = "db modified after last vectorisation"
                info["meta_mtime"] = meta_path.stat().st_mtime if meta_path.exists() else 0.0
                pending[name] = info
                continue

            meta_exists = meta_path.exists()
            meta_mtime = meta_path.stat().st_mtime if meta_exists else 0.0
            info["meta_mtime"] = meta_mtime
            if meta_mtime < db_mtime:
                reason = (
                    "embedding metadata missing" if not meta_exists else "embedding metadata stale"
                )
                info["reason"] = reason
                pending[name] = info
                continue

            if cls:
                try:
                    try:
                        db = cls(vector_backend="annoy")  # type: ignore[call-arg]
                    except Exception:
                        db = cls()  # type: ignore[call-arg]
                    record_count = sum(1 for _ in db.iter_records())
                    vector_count = len(getattr(db, "_metadata", {}))
                    info["record_count"] = record_count
                    info["vector_count"] = vector_count
                    if record_count != vector_count:
                        info["reason"] = (
                            f"record/vector count mismatch {record_count}/{vector_count}"
                        )
                        pending[name] = info
                except Exception:
                    pass
        return pending

    diagnostics = _needs_backfill(names)
    if not diagnostics:
        now = time.time()
        for name in names:
            timestamps[name] = now
        _store_timestamps(timestamps)
        return {} if return_details else None

    for db_name, info in diagnostics.items():
        logger.info(
            "embedding check %s: db_mtime=%s meta_mtime=%s last_vectorization=%s record_count=%s vector_count=%s reason=%s",
            db_name,
            info.get("db_mtime"),
            info.get("meta_mtime"),
            info.get("last_vectorization"),
            info.get("record_count"),
            info.get("vector_count"),
            info.get("reason"),
        )

    if log_hook:
        try:
            log_hook(diagnostics)
        except Exception:  # pragma: no cover - monitoring hooks should not fail
            pass

    pending = diagnostics
    for _ in range(max(retries, 1)):
        asyncio.run(schedule_backfill(dbs=list(pending.keys())))
        time.sleep(delay)
        pending = _needs_backfill(pending.keys())
        if not pending:
            now = time.time()
            for name in names:
                timestamps[name] = now
            _store_timestamps(timestamps)
            return diagnostics if return_details else None

    logger.error(
        "embeddings stale after backfill attempts: %s",
        ", ".join(f"{n} ({r['reason']})" for n, r in pending.items()),
    )
    raise StaleEmbeddingsError({n: r["reason"] for n, r in pending.items()})


__all__ = [
    "EmbeddingBackfill",
    "EmbeddableDBMixin",
    "schedule_backfill",
    "ensure_embeddings_fresh",
    "StaleEmbeddingsError",
    "KNOWN_DB_KINDS",
    "check_staleness",
    "watch_databases",
    "watch_event_bus",
    "consume_record_changes",
]


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - CLI entrypoint
    import argparse

    parser = argparse.ArgumentParser(description="Embedding backfill utility")
    parser.add_argument("--watch", action="store_true", help="run in daemon mode")
    parser.add_argument(
        "--interval", type=float, default=60.0, help="polling interval in seconds"
    )
    parser.add_argument(
        "--db", dest="dbs", action="append", help="database name to process; can repeat"
    )
    parser.add_argument(
        "--verify", action="store_true", help="validate registry and exit"
    )
    args = parser.parse_args(argv)

    eb = EmbeddingBackfill()
    if args.verify:
        eb._verify_registry(args.dbs)
    elif args.watch:
        with eb.watch(interval=args.interval, dbs=args.dbs):
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
    else:
        eb.run(dbs=args.dbs)


if __name__ == "__main__":
    main()
