from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List

from db_router import DBRouter, LOCAL_TABLES
from llm_interface import Prompt, LLMResult

# Ensure the prompts table is treated as local
LOCAL_TABLES.add("prompts")


class PromptDB:
    """Lightweight SQLite logger for LLM prompts."""

    def __init__(self, model: str, path: str | None = None, router: DBRouter | None = None) -> None:
        self.model = model
        db_path = Path(path or os.getenv("PROMPT_DB_PATH", "prompts.db"))
        self.router = router or DBRouter("prompts", str(db_path), str(db_path))
        self.conn = self.router.get_connection("prompts", operation="write")
        self._init_db()

    def _init_db(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS prompts(
                id INTEGER PRIMARY KEY,
                text TEXT,
                examples TEXT,
                vector_confidences TEXT,
                outcome_tags TEXT,
                response_raw TEXT,
                response_text TEXT,
                model TEXT,
                timestamp TEXT
            )
            """,
        )
        self.conn.commit()

    def log_prompt(
        self,
        prompt: Prompt,
        result: LLMResult,
        outcome_tags: List[str],
        vector_confidences: List[float],
    ) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO prompts(
                text, examples, vector_confidences, outcome_tags, response_raw,
                response_text, model, timestamp
            )
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (
                prompt.text,
                json.dumps(prompt.examples),
                json.dumps(vector_confidences),
                json.dumps(outcome_tags),
                json.dumps(result.raw),
                result.text,
                self.model,
                datetime.utcnow().isoformat(),
            ),
        )
        self.conn.commit()
