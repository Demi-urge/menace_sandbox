import json
import sqlite3
from pathlib import Path
from typing import Iterable

from module_index_db import ModuleIndexDB  # type: ignore


class RelevancyMetricsDB:
    """Lightweight store for module invocation metrics."""

    def __init__(self, path: Path | str) -> None:
        self.path = str(path)
        with sqlite3.connect(self.path, check_same_thread=False) as conn:
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
                pass
        with sqlite3.connect(self.path, check_same_thread=False) as conn:
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
