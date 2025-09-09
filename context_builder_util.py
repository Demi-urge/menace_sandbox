from importlib.machinery import SourceFileLoader
from pathlib import Path
import logging

_create_module = SourceFileLoader(
    "config.create_context_builder",
    str(Path(__file__).resolve().parent / "config" / "create_context_builder.py"),
).load_module()

create_context_builder = _create_module.create_context_builder


def ensure_fresh_weights(builder) -> None:
    """Refresh context builder weights with basic error handling."""
    try:
        builder.refresh_db_weights()
    except Exception as exc:  # pragma: no cover - simple wrapper
        logging.getLogger(__name__).warning("refresh_db_weights failed: %s", exc)
        raise
