from __future__ import annotations

"""Utilities for backfilling vector embeddings across databases."""

import logging
import time
from typing import List, Sequence
import importlib
import asyncio
from pathlib import Path
import json
import sys
import hashlib
import pkgutil
import queue

from . import registry as _registry

from .vectorizer import SharedVectorService

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

try:  # pragma: no cover - optional dependency
    from embeddable_db_mixin import EmbeddableDBMixin  # type: ignore
except Exception:  # pragma: no cover
    EmbeddableDBMixin = object  # type: ignore

# Registry describing databases capable of embedding backfills. The file
# ``embedding_registry.json`` lives alongside this module and maps a short name
# to a ``module`` and ``class`` implementing :class:`EmbeddableDBMixin`.
DEFAULT_REGISTRY = resolve_path("vector_service/embedding_registry.json")
_REGISTRY_FILE = DEFAULT_REGISTRY

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
        original_add = getattr(db, "add_embedding", None)
        skipped: List[tuple[str, str]] = []

        if callable(original_add):
            def wrapped_add(record_id, record, kind, *, source_id=""):
                text = record if isinstance(record, str) else str(record)
                lic = license_check(text)
                if lic:
                    _log_violation(
                        str(record_id),
                        lic,
                        license_fingerprint(text),
                    )
                    _RUN_SKIPPED.labels(db.__class__.__name__, lic).inc()
                    skipped.append((str(record_id), lic))
                    return
                return original_add(record_id, record, kind, source_id=source_id)

            db.add_embedding = wrapped_add  # type: ignore[attr-defined]

        try:
            db.backfill_embeddings(batch_size=batch_size)  # type: ignore[call-arg]
        except TypeError:
            db.backfill_embeddings()  # type: ignore[call-arg]
        return skipped

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
    ) -> None:
        """Continuously monitor databases for new or modified records.

        This method is a thin wrapper around :func:`watch_databases` to retain a
        backwards compatible API while the implementation lives at module
        scope.  State is held in memory so restarting the process triggers a
        full scan again.
        """

        watch_databases(interval=interval, dbs=dbs, backend=self.backend)

    def watch_events(
        self,
        *,
        bus: "UnifiedEventBus" | None = None,
        batch_size: int | None = None,
    ) -> None:
        """Listen for database change events and trigger incremental backfills."""

        watch_event_bus(
            bus=bus,
            backend=self.backend,
            batch_size=batch_size if batch_size is not None else 1,
        )


def watch_event_bus(
    *,
    bus: "UnifiedEventBus" | None = None,
    backend: str = "annoy",
    batch_size: int = 1,
) -> None:
    """Listen for database change events via :class:`UnifiedEventBus`."""

    if UnifiedEventBus is None:
        raise RuntimeError("UnifiedEventBus unavailable")

    bus = bus or UnifiedEventBus()
    backfill = EmbeddingBackfill(batch_size=batch_size, backend=backend)
    q: queue.Queue[str] = queue.Queue()
    pending: set[str] = set()

    def _enqueue(kind: str) -> None:
        if kind not in pending:
            pending.add(kind)
            q.put(kind)

    def _handle(_topic: str, event: object) -> None:
        kind = None
        if isinstance(event, dict):
            kind = event.get("db") or event.get("kind")
        elif isinstance(event, str):
            kind = event
        if kind:
            _enqueue(str(kind))

    for topic in ("db:record_added", "db:record_updated", "embedding:backfill"):
        bus.subscribe(topic, _handle)

    while True:
        kind = q.get()
        try:
            backfill.run(
                dbs=[kind],
                batch_size=batch_size,
                backend=backend,
                trigger="event",
            )
        except Exception:  # pragma: no cover - best effort
            logging.getLogger(__name__).exception(
                "event-triggered backfill failed for %s", kind
            )
        finally:
            pending.discard(kind)


def watch_databases(
    *,
    interval: float = 60.0,
    dbs: List[str] | None = None,
    backend: str = "annoy",
) -> None:
    """Continuously monitor databases for new or modified records.

    Every ``interval`` seconds each registered database from
    ``embedding_registry.json`` is scanned using its :meth:`iter_records`
    implementation. Records whose content has not been seen before are
    vectorised via :meth:`SharedVectorService.vectorise_and_store`.
    """

    svc = SharedVectorService()
    seen: dict[str, dict[str, str]] = {}
    backfill = EmbeddingBackfill(backend=backend)
    while True:
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
                except Exception:  # pragma: no cover - best effort
                    logging.getLogger(__name__).exception(
                        "failed to vectorise record %s from %s", rid, key
                    )
        time.sleep(interval)


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


__all__ = [
    "EmbeddingBackfill",
    "EmbeddableDBMixin",
    "schedule_backfill",
    "KNOWN_DB_KINDS",
    "watch_databases",
    "watch_event_bus",
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
        eb.watch(interval=args.interval, dbs=args.dbs)
    else:
        eb.run(dbs=args.dbs)


if __name__ == "__main__":
    main()

