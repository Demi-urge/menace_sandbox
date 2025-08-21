from __future__ import annotations

"""Utilities for backfilling vector embeddings across databases."""

import logging
import time
from typing import List, Sequence
import importlib
import importlib.util
import asyncio
from pathlib import Path
import pkgutil
import json
import sys

from .decorators import log_and_measure
from compliance.license_fingerprint import (
    check as license_check,
    fingerprint as license_fingerprint,
)

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

_RUN_OUTCOME = _me.Gauge(
    "embedding_backfill_runs_total",
    "Outcomes of EmbeddingBackfill.run calls",
    labelnames=["status"],
)
_RUN_DURATION = _me.Gauge(
    "embedding_backfill_run_duration_seconds",
    "Duration of EmbeddingBackfill.run calls",
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
_REGISTRY_FILE = Path(__file__).with_name("embedding_registry.json")

# Minimum set of database kinds expected to support embeddings. These are
# used by :func:`_verify_registry` and other modules to recognise valid
# backfill targets.
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
}


def _load_registry(path: Path | None = None) -> dict[str, tuple[str, str]]:
    """Return mapping of source name to (module, class) tuples.

    The registry is stored as JSON so additional databases can be added without
    modifying code.  Invalid or missing entries are ignored.
    """

    reg_path = path or _REGISTRY_FILE
    try:
        data = json.loads(reg_path.read_text())
    except Exception:  # pragma: no cover - best effort
        return {}
    mapping: dict[str, tuple[str, str]] = {}
    for key, value in data.items():
        mod = value.get("module")
        cls = value.get("class")
        if isinstance(mod, str) and isinstance(cls, str):
            mapping[key] = (mod, cls)
    return mapping


class EmbeddingBackfill:
    """Trigger embedding backfills on all known database classes."""

    def __init__(self, batch_size: int = 100, backend: str = "annoy") -> None:
        self.batch_size = batch_size
        self.backend = backend

    # ------------------------------------------------------------------
    def _load_known_dbs(self, names: List[str] | None = None) -> List[type]:
        """Import all ``EmbeddableDBMixin`` subclasses dynamically.

        The repository is scanned for Python modules referencing
        :class:`EmbeddableDBMixin`.  Any classes found to inherit from the mixin
        are returned.  When ``names`` is provided the result is filtered to
        include only classes whose name matches any entry. Matching is
        case-insensitive and ignores plural forms or a trailing ``DB`` suffix.
        """

        subclasses: List[type] = []

        root = Path(__file__).resolve().parents[1]
        for mod in pkgutil.walk_packages([str(root)]):  # pragma: no cover - best effort
            name = mod.name
            if any(part in {"tests", "scripts", "docs"} for part in name.split(".")):
                continue
            try:
                spec = importlib.util.find_spec(name)
                if not spec or not spec.origin or not spec.origin.endswith(".py"):
                    continue
                path = Path(spec.origin)
                if "EmbeddableDBMixin" not in path.read_text(encoding="utf-8"):
                    continue
                importlib.import_module(name)
            except Exception:
                continue

        # Ensure explicit modules from the registry are present even if discovery
        # misses them.  Loading happens dynamically so the registry can be
        # extended without code changes.
        for mod_name, cls_name in _load_registry().values():
            try:
                mod = importlib.import_module(mod_name)
                getattr(mod, cls_name)
            except Exception:
                continue

        try:
            subclasses = [
                cls
                for cls in EmbeddableDBMixin.__subclasses__()
                if hasattr(cls, "backfill_embeddings")
            ]
        except Exception:  # pragma: no cover - defensive
            subclasses = []

        # Deduplicate while preserving order
        seen: set[type] = set()
        unique: List[type] = []
        for cls in subclasses:
            if cls not in seen:
                seen.add(cls)
                unique.append(cls)
        subclasses = unique

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
        # ensure discovered classes implement the expected interface
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
        pkg_root_path = Path(__file__).resolve().parents[1]
        pkg_root = pkg_root_path.name
        parent = pkg_root_path.parent
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        to_check = set(names or registry.keys())
        for name in to_check:
            mod_cls = registry.get(name)
            if not mod_cls:
                if names is None and _REGISTRY_FILE == Path(__file__).with_name("embedding_registry.json"):
                    problems.append(f"{name}: not registered")
                continue
            mod_name, cls_name = mod_cls
            try:
                mod = importlib.import_module(mod_name)
            except Exception:
                try:
                    mod = importlib.import_module(f"{pkg_root}.{mod_name}")
                except Exception as exc:  # pragma: no cover - defensive
                    problems.append(f"{name}: import failed ({exc})")
                    continue
            try:
                cls = getattr(mod, cls_name)
            except Exception as exc:  # pragma: no cover - defensive
                problems.append(f"{name}: import failed ({exc})")
                continue
            if not issubclass(cls, EmbeddableDBMixin):
                problems.append(f"{name}: not EmbeddableDBMixin")
                continue
            missing = [m for m in ("iter_records", "vector") if not callable(getattr(cls, m, None))]
            if missing:
                problems.append(f"{name}: missing {', '.join(missing)}")
        if names is None and _REGISTRY_FILE == Path(__file__).with_name("embedding_registry.json"):
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
            _RUN_OUTCOME.labels(status).inc()
            _RUN_DURATION.set(time.time() - start)
            raise
        _RUN_OUTCOME.labels(status).inc()
        _RUN_DURATION.set(time.time() - start)


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

    subclasses = backfill._load_known_dbs(names=list(dbs) if dbs else None)

    async def _run(cls: type) -> None:
        await asyncio.to_thread(backfill.run, db=cls.__name__)

    await asyncio.gather(*[_run(cls) for cls in subclasses])


__all__ = ["EmbeddingBackfill", "EmbeddableDBMixin", "schedule_backfill", "KNOWN_DB_KINDS"]

