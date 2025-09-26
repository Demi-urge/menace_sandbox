from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from config import ContextBuilderConfig, StackDatasetConfig, get_config
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


def _resolve_stack_path(value: Any, base_dir: Path) -> Path | None:
    if value in {None, "", b""}:
        return None
    try:
        candidate = Path(str(value))
    except Exception:
        return None
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    return candidate


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
        cfg = get_config()
        context_cfg = getattr(cfg, "context_builder", ContextBuilderConfig())
    except Exception:
        context_cfg = ContextBuilderConfig()

    stack_cfg: StackDatasetConfig = getattr(context_cfg, "stack", StackDatasetConfig())
    stack_enabled = bool(getattr(stack_cfg, "enabled", False))

    env_index = os.getenv("STACK_INDEX_PATH")
    env_metadata = os.getenv("STACK_METADATA_PATH")
    env_cache = os.getenv("STACK_CACHE_DIR")
    env_progress = os.getenv("STACK_PROGRESS_PATH")

    index_path = _resolve_stack_path(env_index, data_dir) or _resolve_stack_path(
        getattr(stack_cfg, "index_path", None), data_dir
    )
    metadata_path = _resolve_stack_path(env_metadata, data_dir) or _resolve_stack_path(
        getattr(stack_cfg, "metadata_path", None), data_dir
    )
    cache_dir = _resolve_stack_path(env_cache, data_dir) or _resolve_stack_path(
        getattr(stack_cfg, "cache_dir", None), data_dir
    )
    progress_path = _resolve_stack_path(env_progress, data_dir) or _resolve_stack_path(
        getattr(stack_cfg, "progress_path", None), data_dir
    )

    stack_kwargs: Dict[str, Any] = {
        "config": context_cfg,
        "stack_config": stack_cfg,
    }

    if stack_enabled:
        if index_path is None:
            raise FileNotFoundError(
                "Stack retrieval enabled but no index_path configured. "
                "Set context_builder.stack.index_path or STACK_INDEX_PATH."
            )
        if not index_path.exists():
            raise FileNotFoundError(
                f"Stack index expected at {index_path} but was not found"
            )
        stack_kwargs["stack_index_path"] = str(index_path)

        if metadata_path is None:
            raise FileNotFoundError(
                "Stack retrieval enabled but no metadata_path configured. "
                "Set context_builder.stack.metadata_path or STACK_METADATA_PATH."
            )
        metadata_str = _ensure_readable(metadata_path, "stack metadata store")
        stack_kwargs["stack_metadata_path"] = metadata_str
    else:
        if index_path is not None:
            stack_kwargs["stack_index_path"] = str(index_path)
        if metadata_path is not None:
            stack_kwargs["stack_metadata_path"] = str(metadata_path)

    if cache_dir is not None:
        stack_kwargs["stack_cache_dir"] = str(cache_dir)
    if progress_path is not None:
        stack_kwargs["stack_progress_path"] = str(progress_path)

    try:
        init_kwargs: Dict[str, Any] = dict(builder_kwargs)
        init_kwargs.update(stack_kwargs)
        builder = ContextBuilder(**init_kwargs)
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
