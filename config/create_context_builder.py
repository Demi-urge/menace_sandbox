from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Dict, Iterable, Tuple

from vector_service.context_builder import ContextBuilder

DB_SPEC: Tuple[Tuple[str, str, str], ...] = (
    ("bots_db", "bots.db", "BOT_DB_PATH"),
    ("code_db", "code.db", "CODE_DB_PATH"),
    ("errors_db", "errors.db", "ERROR_DB_PATH"),
    ("workflows_db", "workflows.db", "WORKFLOW_DB_PATH"),
)


def _resolve_db_paths(data_dir: Path) -> Iterable[Tuple[str, str, Path]]:
    for attr, filename, env_var in DB_SPEC:
        env_value = os.getenv(env_var)
        path = Path(env_value) if env_value else data_dir / filename
        yield attr, filename, path


def _ensure_readable(path: Path, filename: str) -> str:
    try:
        with path.open("rb"):
            pass
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Path for '{filename}' does not exist: {path}"
        ) from exc
    except OSError as exc:
        raise OSError(f"Path for '{filename}' is not readable: {path}") from exc
    return str(path)


def create_context_builder() -> ContextBuilder:
    """Return a :class:`ContextBuilder` wired to the standard local databases.

    All four database paths (``bots.db``, ``code.db``, ``errors.db`` and
    ``workflows.db``) are mandatory and must exist and be readable before
    instantiating :class:`ContextBuilder`.
    """

    data_dir = Path(os.getenv("SANDBOX_DATA_DIR", "."))
    builder_kwargs: Dict[str, str] = {}
    for attr, filename, path in _resolve_db_paths(data_dir):
        builder_kwargs[attr] = _ensure_readable(path, filename)

    try:
        builder = ContextBuilder(**builder_kwargs)
    except TypeError as exc:  # pragma: no cover - for simple stubs in tests
        raise ValueError(
            "ContextBuilder requires paths to 'bots.db', 'code.db', 'errors.db', "
            "and 'workflows.db'"
        ) from exc
    try:
        ensure_stack = getattr(builder, "ensure_stack_embeddings", None)
        if callable(ensure_stack):
            ensure_stack()
    except Exception as exc:  # pragma: no cover - best effort warning
        logging.getLogger(__name__).warning(
            "Stack ingestion check failed: %s", exc
        )
    return builder
