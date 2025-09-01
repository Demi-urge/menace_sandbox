from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List

from db_router import DBRouter, LOCAL_TABLES
from llm_interface import Completion, Prompt

# Ensure the prompts table is treated as local
LOCAL_TABLES.add("prompts")


class PromptDB:
    """Lightweight SQLite logger for LLM prompts and completions."""

    def __init__(
        self, model: str, path: str | None = None, router: DBRouter | None = None
    ) -> None:
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
                response_parsed TEXT,
                response_text TEXT,
                model TEXT,
                timestamp TEXT
            )
            """,
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    def log(self, prompt: Prompt, result: Completion) -> None:
        """Persist *prompt* and *result* to the underlying SQLite store."""

        tags = (
            prompt.outcome_tags
            or prompt.metadata.get("tags")
            or prompt.metadata.get("outcome_tags")
            or []
        )
        confs = (
            prompt.vector_confidences
            or prompt.metadata.get("vector_confidences")
            or []
        )
        if not isinstance(tags, list):
            tags = [str(tags)]
        try:
            confs = [float(c) for c in confs]
        except Exception:  # pragma: no cover - defensive
            confs = []
        try:
            parsed = json.dumps(result.parsed)
        except Exception:  # pragma: no cover - defensive
            parsed = json.dumps(None)

        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO prompts(
                text, examples, vector_confidences, outcome_tags, response_raw,
                response_parsed, response_text, model, timestamp
            )
            VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                prompt.text,
                json.dumps(prompt.examples),
                json.dumps(confs),
                json.dumps(tags),
                json.dumps(result.raw),
                parsed,
                result.text,
                self.model,
                datetime.utcnow().isoformat(),
            ),
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    def log_prompt(
        self,
        prompt: Prompt,
        result: Completion,
        outcome_tags: List[str] | None = None,
        vector_confidences: List[float] | None = None,
    ) -> None:
        """Backward compatible wrapper around :meth:`log`."""

        if outcome_tags is not None:
            prompt.outcome_tags = list(outcome_tags)
        if vector_confidences is not None:
            prompt.vector_confidences = list(vector_confidences)
        self.log(prompt, result)
