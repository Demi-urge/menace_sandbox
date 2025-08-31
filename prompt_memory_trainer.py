import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Any


class PromptMemoryTrainer:
    """Persist and analyse prompt formatting metadata.

    Records from :class:`PromptEngine` and :class:`SelfCodingEngine` are stored
    in a lightweight SQLite database.  Each entry captures the tone used for a
    prompt, the headers inserted and the order of examples alongside a boolean
    flag indicating whether the resulting patch succeeded.
    """

    def __init__(self, db_path: str | Path = "prompt_format_history.db") -> None:
        self.db_path = Path(db_path)
        self._init_db()

    # ------------------------------------------------------------------
    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prompt_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tone TEXT,
                    headers TEXT,
                    example_order TEXT,
                    success INTEGER
                )
                """
            )

    # ------------------------------------------------------------------
    def record(
        self,
        *,
        tone: str,
        headers: List[str],
        example_order: List[str],
        success: bool,
    ) -> None:
        """Store a single prompt formatting outcome."""

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO prompt_history (tone, headers, example_order, success)"
                " VALUES (?, ?, ?, ?)",
                (
                    tone,
                    json.dumps(headers),
                    json.dumps(example_order),
                    1 if success else 0,
                ),
            )

    # ------------------------------------------------------------------
    def train(self) -> Dict[str, Dict[str, float]]:
        """Return success rates aggregated by formatting parameter."""

        summary: Dict[str, Dict[str, float]] = {}
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            for column in ("tone", "headers", "example_order"):
                cur.execute(
                    f"SELECT {column}, AVG(success) FROM prompt_history GROUP BY {column}"
                )
                summary[column] = {
                    row[0]: float(row[1] or 0.0) for row in cur.fetchall()
                }
        return summary


__all__ = ["PromptMemoryTrainer"]
