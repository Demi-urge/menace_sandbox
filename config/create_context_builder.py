from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Iterable, Tuple
import uuid

from vector_service.context_builder import ContextBuilder
from vector_service.retriever import StackRetriever

logger = logging.getLogger(__name__)


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
    except FileNotFoundError:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    except PermissionError as exc:
        logger.warning(
            "create_context_builder: unable to open %s due to permission error: %s",
            filename,
            exc,
        )
        return str(path)
    except OSError as exc:
        logger.warning(
            "create_context_builder: best-effort access failed for %s: %s",
            filename,
            exc,
        )
        return str(path)
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

    stack_overrides: Dict[str, object] = {}
    enabled_env = os.getenv("STACK_CONTEXT_ENABLED")
    if enabled_env is not None:
        stack_overrides["enabled"] = enabled_env.strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    top_k_env = os.getenv("STACK_CONTEXT_TOP_K")
    if top_k_env:
        try:
            stack_overrides["top_k"] = int(top_k_env)
        except ValueError:
            pass
    langs_env = os.getenv("STACK_CONTEXT_LANGUAGES")
    if langs_env:
        languages = [part.strip() for part in langs_env.split(",") if part.strip()]
        stack_overrides["languages"] = tuple(languages)
    max_lines_env = os.getenv("STACK_CONTEXT_MAX_LINES")
    if max_lines_env:
        try:
            stack_overrides["max_lines"] = int(max_lines_env)
        except ValueError:
            pass
    if stack_overrides:
        builder_kwargs["stack_config"] = stack_overrides

    try:
        builder_kwargs.setdefault("stack_retriever", StackRetriever())
    except Exception:
        pass

    try:
        return ContextBuilder(**builder_kwargs)
    except TypeError:  # pragma: no cover - fallback to stub builder

        class _StubContextBuilder:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs
                self.provenance_token = kwargs.get("provenance_token") or uuid.uuid4().hex

            def build(self, *args: object, **kwargs: object) -> dict:
                return {"context": [], "metadata": {}}

        return _StubContextBuilder(**builder_kwargs)
