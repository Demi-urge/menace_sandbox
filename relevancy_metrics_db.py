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
                    tags TEXT
                )
                """
            )

    # --------------------------------------------------------------
    def record(
        self,
        module: str,
        elapsed: float,
        module_index: ModuleIndexDB | None = None,
        tags: Iterable[str] | None = None,
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
                "SELECT call_count, total_time, tags FROM module_metrics WHERE module_id=?",
                (mid,),
            ).fetchone()
            if row:
                count, total, existing = row
                count = int(count) + 1
                total = float(total) + float(elapsed)
                current = set(json.loads(existing) if existing else [])
                current.update(tag_set)
                conn.execute(
                    "UPDATE module_metrics SET module_name=?, call_count=?, total_time=?, tags=? WHERE module_id=?",
                    (
                        module,
                        count,
                        total,
                        json.dumps(sorted(current)),
                        mid,
                    ),
                )
            else:
                conn.execute(
                    "INSERT INTO module_metrics(module_id, module_name, call_count, total_time, tags) VALUES (?, ?, 1, ?, ?)",
                    (mid, module, float(elapsed), json.dumps(sorted(tag_set))),
                )
            conn.commit()
