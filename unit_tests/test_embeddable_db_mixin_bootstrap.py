from __future__ import annotations

import importlib
import sys
from typing import Dict
from unittest import mock


_MODULE_ALIASES = (
    "embeddable_db_mixin",
    "menace_sandbox.embeddable_db_mixin",
)


def _snapshot_and_clear() -> Dict[str, object]:
    """Remove mixin aliases while returning any prior modules."""

    existing: Dict[str, object] = {}
    for name in _MODULE_ALIASES:
        module = sys.modules.get(name)
        if module is not None:
            existing[name] = module
        sys.modules.pop(name, None)
    return existing


def _restore_modules(existing: Dict[str, object]) -> None:
    """Restore module aliases after a test run."""

    for name in _MODULE_ALIASES:
        sys.modules.pop(name, None)
    for name, module in existing.items():
        sys.modules[name] = module


def _exercise_import_sequence(first: str, second: str) -> None:
    """Import the mixin using *first* and *second* module paths."""

    previous = _snapshot_and_clear()
    try:
        with (
            mock.patch("menace_sandbox.vector_metrics_db.VectorMetricsDB") as mock_vec_db,
            mock.patch("menace_sandbox.embedding_stats_db.EmbeddingStatsDB") as mock_stats_db,
        ):
            module_first = importlib.import_module(first)
            module_second = importlib.import_module(second)

            assert module_first is module_second
            assert module_first._VEC_METRICS is mock_vec_db.return_value
            assert module_first._EMBED_STATS_DB is mock_stats_db.return_value
            assert mock_vec_db.call_count == 1
            assert mock_stats_db.call_count == 1
    finally:
        _restore_modules(previous)


def test_embeddable_db_mixin_bootstrap_aliases_and_singletons() -> None:
    """The mixin must load identically via package and flat import paths."""

    _exercise_import_sequence("embeddable_db_mixin", "menace_sandbox.embeddable_db_mixin")
    _exercise_import_sequence("menace_sandbox.embeddable_db_mixin", "embeddable_db_mixin")
