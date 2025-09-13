from __future__ import annotations

import os
from pathlib import Path

from vector_service.context_builder import ContextBuilder


def create_context_builder() -> ContextBuilder:
    """Return a :class:`ContextBuilder` wired to the standard local databases."""
    data_dir = Path(os.getenv("SANDBOX_DATA_DIR", "."))
    bot_db = Path(os.getenv("BOT_DB_PATH", data_dir / "bots.db"))
    code_db = Path(os.getenv("CODE_DB_PATH", data_dir / "code.db"))
    error_db = Path(os.getenv("ERROR_DB_PATH", data_dir / "errors.db"))
    workflow_db = Path(os.getenv("WORKFLOW_DB_PATH", data_dir / "workflows.db"))
    try:
        return ContextBuilder(
            str(bot_db), str(code_db), str(error_db), str(workflow_db)
        )
    except TypeError as exc:  # pragma: no cover - for simple stubs in tests
        raise ValueError(
            "ContextBuilder requires paths to 'bots.db', 'code.db', 'errors.db', "
            "and 'workflows.db'"
        ) from exc
