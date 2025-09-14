"""SQLite database management utilities for model tracking."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from db_router import GLOBAL_ROUTER
from dynamic_path_router import resolve_path

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - for type hints
    from .preliminary_research_bot import PreliminaryResearchBot
else:  # pragma: no cover - lightweight fallback implementation
    from .preliminary_research_bot import BusinessData
    from .bot_registry import BotRegistry
    from .coding_bot_interface import self_coding_managed
    from .data_bot import DataBot
    from .self_coding_manager import SelfCodingManager

    bot_registry = BotRegistry()
    data_bot = DataBot(start_server=False)

    @self_coding_managed(
        bot_registry=bot_registry, data_bot=data_bot, manager=SelfCodingManager
    )
    class PreliminaryResearchBot:
        """Minimal fallback that attempts a basic scrape for metrics."""

        def _extract(self, text: str, term: str) -> float | None:
            import re
            m = re.search(rf"{term}[^0-9]*(\d+(?:\.\d+)?)", text, flags=re.I)
            return float(m.group(1)) if m else None

        def process_model(self, name: str, urls: Iterable[str]) -> BusinessData:
            try:
                from .preliminary_research_bot import PreliminaryResearchBot as _Real
                return _Real().process_model(name, urls)
            except Exception:
                texts: List[str] = []
                try:
                    import requests  # type: ignore
                except Exception:
                    requests = None  # type: ignore
                    from urllib.request import urlopen
                for url in urls:
                    try:
                        if requests:
                            resp = requests.get(url, timeout=5)
                            texts.append(resp.text)
                        else:
                            with urlopen(url, timeout=5) as fh:
                                texts.append(fh.read().decode())
                    except Exception:
                        continue
                combined = " ".join(texts)
                return BusinessData(
                    model_name=name,
                    profit_margin=self._extract(combined, "profit margin"),
                    operational_cost=self._extract(combined, "operational cost"),
                    market_saturation=self._extract(combined, "market saturation"),
                    competitors=[],
                    keywords=[],
                    roi_score=None,
                )

DB_PATH = resolve_path("models.db")


@contextmanager
def get_connection(
    db_path: Path = DB_PATH, *, conn: sqlite3.Connection | None = None
) -> Iterable[sqlite3.Connection]:
    """Context manager yielding a SQLite connection.

    If ``conn`` is provided it is assumed to be a pre-routed connection and will
    be used directly. Otherwise a connection is obtained from the global router.
    The context manager ensures commits and rollbacks mirror the behaviour of
    :func:`sqlite3.connect` when used as a context manager.
    """

    if conn is None:
        if GLOBAL_ROUTER is None:
            raise RuntimeError("Database router is not initialised")
        with GLOBAL_ROUTER.get_connection("models") as conn:
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
    else:
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def init_db(conn: sqlite3.Connection) -> None:
    """Create tables if they do not already exist."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE COLLATE NOCASE,
            source TEXT,
            date_discovered TEXT,
            tags TEXT,
            roi_metadata TEXT,
            exploration_status TEXT,
            profitability_score REAL DEFAULT 0,
            current_roi REAL DEFAULT 0,
            final_roi_prediction REAL DEFAULT 0,
            initial_roi_prediction REAL DEFAULT 0,
            current_status TEXT,
            workflow_id INTEGER
        )
        """
    )
    cols = [r[1] for r in conn.execute("PRAGMA table_info(models)").fetchall()]
    if "profitability_score" not in cols:
        conn.execute("ALTER TABLE models ADD COLUMN profitability_score REAL DEFAULT 0")
    if "current_roi" not in cols:
        conn.execute("ALTER TABLE models ADD COLUMN current_roi REAL DEFAULT 0")
    if "final_roi_prediction" not in cols:
        conn.execute("ALTER TABLE models ADD COLUMN final_roi_prediction REAL DEFAULT 0")
    if "initial_roi_prediction" not in cols:
        conn.execute("ALTER TABLE models ADD COLUMN initial_roi_prediction REAL DEFAULT 0")
    if "current_status" not in cols:
        conn.execute("ALTER TABLE models ADD COLUMN current_status TEXT")
    if "workflow_id" not in cols:
        conn.execute("ALTER TABLE models ADD COLUMN workflow_id INTEGER")


def add_model(
    name: str,
    source: str = "",
    tags: str = "",
    roi_metadata: str = "",
    exploration_status: str = "",
    profitability_score: float = 0.0,
    current_roi: float = 0.0,
    final_roi_prediction: float = 0.0,
    initial_roi_prediction: float = 0.0,
    current_status: str = "",
    workflow_id: int | None = None,
    db_path: Path = DB_PATH,
) -> int:
    """Insert a new model if it does not already exist.

    Returns the model id from the database.
    """
    with get_connection(db_path) as conn:
        init_db(conn)
        cur = conn.execute(
            "SELECT id FROM models WHERE name LIKE ?",
            (f"%{name}%",),
        )
        row = cur.fetchone()
        if row:
            return row[0]
        cur = conn.execute(
            """
            INSERT INTO models (
                name,
                source,
                date_discovered,
                tags,
                roi_metadata,
                exploration_status,
                profitability_score,
                current_roi,
                final_roi_prediction,
                initial_roi_prediction,
                current_status,
                workflow_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                source,
                datetime.utcnow().isoformat(timespec="seconds"),
                tags,
                roi_metadata,
                exploration_status,
                profitability_score,
                current_roi,
                final_roi_prediction,
                initial_roi_prediction,
                current_status,
                workflow_id,
            ),
        )
        return cur.lastrowid


def search_models(
    keyword: str | None = None,
    *,
    tags: Optional[List[str]] = None,
    limit: int = 10,
    db_path: Path = DB_PATH,
) -> List[Dict[str, Any]]:
    """Search models by keyword and/or tags.

    Returns a list of result dictionaries ordered by a crude confidence score.
    """
    query = (
        "SELECT id, name, source, date_discovered, tags, roi_metadata, "
        "exploration_status, profitability_score, current_roi, final_roi_prediction, "
        "initial_roi_prediction, current_status FROM models"
    )
    criteria: List[str] = []
    params: List[str] = []

    if keyword:
        criteria.append("LOWER(name) LIKE ?")
        params.append(f"%{keyword.lower()}%")
    if tags:
        for t in tags:
            criteria.append("tags LIKE ?")
            params.append(f"%{t}%")

    if criteria:
        query += " WHERE " + " AND ".join(criteria)

    with get_connection(db_path) as conn:
        init_db(conn)
        cur = conn.execute(query, params)
        rows = cur.fetchall()

    def score(row: tuple[str, ...]) -> int:
        text = (row[1] or "") + " " + (row[4] or "")
        score = 0
        if keyword and keyword.lower() in text.lower():
            score += 50
        if tags:
            score += sum(10 for t in tags if t in text)
        return score

    results = [
            {
                "id": row[0],
                "name": row[1],
                "source": row[2],
                "date_discovered": row[3],
                "tags": row[4],
                "roi_metadata": row[5],
                "exploration_status": row[6],
                "profitability_score": row[7],
                "current_roi": row[8],
                "final_roi_prediction": row[9] if len(row) > 9 else 0.0,
                "initial_roi_prediction": row[10] if len(row) > 10 else 0.0,
                "current_status": row[11] if len(row) > 11 else "",
                "score": score(row),
            }
            for row in rows
        ]
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:limit]


def update_model(id: int, db_path: Path = DB_PATH, **fields: Any) -> None:
    """Update fields for a given model id."""
    if not fields:
        return
    assignments = ", ".join(f"{k} = ?" for k in fields)
    params = list(fields.values()) + [id]
    with get_connection(db_path) as conn:
        conn.execute(f"UPDATE models SET {assignments} WHERE id = ?", params)


def delete_model(id: int, db_path: Path = DB_PATH) -> None:
    """Delete a model from the database."""
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM models WHERE id = ?", (id,))


def integrity_check(db_path: Path = DB_PATH) -> bool:
    """Run integrity checks and return True if OK."""
    with get_connection(db_path) as conn:
        ok = conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        duplicates = conn.execute(
            "SELECT name, COUNT(*) FROM models GROUP BY LOWER(name) HAVING COUNT(*) > 1"
        ).fetchall()
        for name, _ in duplicates:
            conn.execute(
                "DELETE FROM models WHERE id NOT IN (SELECT MIN(id) FROM models "
                "WHERE LOWER(name) = LOWER(?) )",
                (name,),
            )
    return ok and not duplicates


def calculate_threshold(energy_score: float, base: float = 70.0) -> float:
    """Return profitability threshold adjusted by energy score."""
    return base - energy_score * 10.0


def compute_profitability_score(data: "BusinessData") -> float:
    """Derive a simple profitability score from research data."""
    margin = data.profit_margin or 0.0
    cost = data.operational_cost or 0.0
    saturation = data.market_saturation or 0.0
    score = margin - cost - saturation
    return max(0.0, min(100.0, score))


def evaluate_candidate(
    name: str,
    urls: Iterable[str],
    prelim: "PreliminaryResearchBot",
    threshold: float,
    db_path: Path = DB_PATH,
) -> str:
    """Process a candidate model name and store it with research results."""
    if search_models(name, db_path=db_path):
        return "killed"
    data = prelim.process_model(name, urls)
    score = compute_profitability_score(data)
    status = "pending" if score >= threshold else "invalid"
    add_model(
        name,
        source="preliminary",
        tags=" ".join(data.keywords),
        roi_metadata="",
        exploration_status=status,
        profitability_score=score,
        current_roi=data.roi_score or 0.0,
        final_roi_prediction=data.roi_score or 0.0,
        initial_roi_prediction=data.roi_score or 0.0,
        current_status=status,
        db_path=db_path,
    )
    return status


def apply_threshold(threshold: float, db_path: Path = DB_PATH) -> None:
    """Update stored models according to a new profitability threshold."""
    with get_connection(db_path) as conn:
        conn.execute(
            "UPDATE models SET exploration_status = 'pending', current_status = 'pending' "
            "WHERE exploration_status = 'invalid' AND profitability_score >= ?",
            (threshold,),
        )


def update_profitability_threshold(energy_score: float, *, db_path: Path = DB_PATH) -> float:
    """Recalculate and apply the profitability threshold.

    Returns the newly computed threshold value.
    """
    threshold = calculate_threshold(energy_score)
    apply_threshold(threshold, db_path=db_path)
    return threshold


def process_idea(
    name: str,
    tags: Iterable[str],
    source: str,
    urls: Iterable[str],
    prelim: "PreliminaryResearchBot",
    energy_score: float,
    *,
    db_path: Path = DB_PATH,
) -> str:
    """Full pipeline for handling a new model idea.

    If the name already exists it is immediately killed. Otherwise the idea is
    sent through ``PreliminaryResearchBot`` and stored with its profitability
    score. The current threshold derived from ``energy_score`` determines if the
    model is greenlit or marked unprofitable. When the threshold changes, any
    previously unprofitable models that now meet the requirement are updated via
    :func:`apply_threshold`.
    """
    if search_models(name, db_path=db_path):
        return "killed"

    threshold = calculate_threshold(energy_score)
    data = prelim.process_model(name, urls)
    score = compute_profitability_score(data)
    status = "pending" if score >= threshold else "invalid"
    add_model(
        name,
        source=source,
        tags=" ".join(tags),
        roi_metadata="",
        exploration_status=status,
        profitability_score=score,
        current_roi=data.roi_score or 0.0,
        final_roi_prediction=data.roi_score or 0.0,
        initial_roi_prediction=data.roi_score or 0.0,
        current_status=status,
        db_path=db_path,
    )
    apply_threshold(threshold, db_path=db_path)
    return status


def submit_idea(
    name: str,
    tags: Iterable[str],
    source: str,
    *,
    db_path: Path = DB_PATH,
) -> str:
    """Receive an idea from ChatGPT Idea Bot and forward it for research."""
    prelim = PreliminaryResearchBot()
    return process_idea(
        name,
        tags=tags,
        source=source,
        urls=[],
        prelim=prelim,
        energy_score=0.0,
        db_path=db_path,
    )
