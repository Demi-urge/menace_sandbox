import json
import logging
import sqlite3
from pathlib import Path
from typing import Iterable, TYPE_CHECKING, Any

from db_router import DBRouter, GLOBAL_ROUTER, LOCAL_TABLES, init_db_router

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - only for type hints
    from module_index_db import ModuleIndexDB  # type: ignore
else:  # pragma: no cover - fallback when module_index_db is unavailable
    ModuleIndexDB = Any  # type: ignore


class RelevancyMetricsDB:
    """Lightweight store for module invocation metrics."""

    def __init__(self, path: Path | str, router: DBRouter | None = None) -> None:
        self.path = str(path)
        LOCAL_TABLES.add("module_metrics")
        self.router = router or GLOBAL_ROUTER or init_db_router(
            "relevancy_metrics_db", local_db_path=self.path, shared_db_path=self.path
        )
        conn = self.router.get_connection("module_metrics")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS module_metrics (
                module_id INTEGER PRIMARY KEY,
                module_name TEXT,
                call_count INTEGER DEFAULT 0,
                total_time REAL DEFAULT 0.0,
                roi_delta REAL DEFAULT 0.0,
                tags TEXT
            )
            """
        )
        try:
            conn.execute(
                "ALTER TABLE module_metrics ADD COLUMN roi_delta REAL DEFAULT 0.0"
            )
        except sqlite3.OperationalError:
            # Column already exists – ignore migration errors
            pass

    # --------------------------------------------------------------
    def record(
        self,
        module: str,
        elapsed: float,
        module_index: ModuleIndexDB | None = None,
        tags: Iterable[str] | None = None,
        roi_delta: float | None = None,
    ) -> None:
        """Update metrics for ``module``.

        ``elapsed`` denotes execution time for the invocation. ``tags`` are
        additional high level tags. When ``module_index`` is provided the module
        identifier and stored tags are fetched from it.
        """

        mid = (
            module_index.get(module) if module_index else abs(hash(module)) % 1000
        )
        tag_set = set(tags or [])
        if module_index:
            try:
                tag_set.update(module_index.get_tags(module))
            except Exception:
                logger.warning(
                    "Failed to fetch tags for %s from module index at %s",
                    module,
                    self.path,
                    exc_info=True,
                )
        conn = self.router.get_connection("module_metrics")
        row = conn.execute(
            "SELECT call_count, total_time, roi_delta, tags FROM module_metrics WHERE module_id=?",
            (mid,),
        ).fetchone()
        if row:
            count, total, roi_val, existing = row
            count = int(count) + 1
            total = float(total) + float(elapsed)
            roi_val = float(roi_val or 0.0) + float(roi_delta or 0.0)
            current = set(json.loads(existing) if existing else [])
            current.update(tag_set)
            conn.execute(
                "UPDATE module_metrics SET module_name=?, call_count=?, total_time=?, roi_delta=?, tags=? WHERE module_id=?",
                (
                    module,
                    count,
                    total,
                    roi_val,
                    json.dumps(sorted(current)),
                    mid,
                ),
            )
        else:
            conn.execute(
                "INSERT INTO module_metrics(module_id, module_name, call_count, total_time, roi_delta, tags) VALUES (?, ?, 1, ?, ?, ?)",
                (
                    mid,
                    module,
                    float(elapsed),
                    float(roi_delta or 0.0),
                    json.dumps(sorted(tag_set)),
                ),
            )
        conn.commit()

    # --------------------------------------------------------------
    def get_roi_deltas(self, modules: Iterable[str] | None = None) -> dict[str, float]:
        """Return cumulative ROI deltas for the given ``modules``.

        When ``modules`` is ``None`` all entries in the database are returned.
        Missing modules default to ``0.0`` and are simply omitted from the
        result mapping.
        """

        query = "SELECT module_name, roi_delta FROM module_metrics"
        params: tuple[str, ...] = ()
        module_list = list(modules or [])
        if module_list:
            placeholders = ",".join("?" for _ in module_list)
            query += f" WHERE module_name IN ({placeholders})"
            params = tuple(module_list)
        conn = self.router.get_connection("module_metrics")
        rows = conn.execute(query, params).fetchall()
        return {str(name): float(delta or 0.0) for name, delta in rows}
