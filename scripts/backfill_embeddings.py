"""CLI for running embedding backfills across databases."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repository root on ``sys.path`` when executed directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

import importlib
from typing import Dict, List, Tuple, Type

try:
    from menace_sandbox.embeddable_db_mixin import EmbeddableDBMixin
except ModuleNotFoundError:  # pragma: no cover - legacy flat import support
    from embeddable_db_mixin import EmbeddableDBMixin
from embedding_stats_db import EmbeddingStatsDB
from vector_service import VectorServiceError


# Map friendly names to the modules and classes implementing
# :class:`EmbeddableDBMixin`. Synonyms map to the same underlying entry.
_DB_REGISTRY: Dict[str, Tuple[str, str]] = {
    "bot": ("bot_database", "BotDB"),
    "bots": ("bot_database", "BotDB"),
    "botdb": ("bot_database", "BotDB"),
    "workflow": ("task_handoff_bot", "WorkflowDB"),
    "workflows": ("task_handoff_bot", "WorkflowDB"),
    "workflowdb": ("task_handoff_bot", "WorkflowDB"),
    "error": ("error_bot", "ErrorDB"),
    "errors": ("error_bot", "ErrorDB"),
    "errordb": ("error_bot", "ErrorDB"),
    "information": ("information_db", "InformationDB"),
    "info": ("information_db", "InformationDB"),
    "infodb": ("information_db", "InformationDB"),
    "enhancement": ("chatgpt_enhancement_bot", "EnhancementDB"),
    "enhancements": ("chatgpt_enhancement_bot", "EnhancementDB"),
    "enhancementdb": ("chatgpt_enhancement_bot", "EnhancementDB"),
    "research": ("research_aggregator_bot", "InfoDB"),
    "researchdb": ("research_aggregator_bot", "InfoDB"),
}


def _resolve_db(name: str) -> Type[EmbeddableDBMixin]:
    """Return the :class:`EmbeddableDBMixin` subclass for ``name``."""

    key = name.lower()
    module_name, class_name = _DB_REGISTRY.get(key, (key, ""))
    module = importlib.import_module(module_name)
    if class_name:
        cls = getattr(module, class_name)
        if not issubclass(cls, EmbeddableDBMixin):
            raise VectorServiceError(f"{class_name} is not embeddable")
        return cls
    for obj in vars(module).values():
        if isinstance(obj, type) and issubclass(obj, EmbeddableDBMixin):
            return obj
    raise VectorServiceError(f"no EmbeddableDBMixin subclass found in {name}")


def _collect_stats(stats: EmbeddingStatsDB, db_name: str) -> Tuple[int, float, float, float, int]:
    """Return aggregated metrics for ``db_name``."""

    cur = stats.conn.execute(
        """
        SELECT COUNT(*), COALESCE(SUM(tokens),0), COALESCE(SUM(wall_ms),0),
               COALESCE(SUM(store_ms),0)
        FROM embedding_stats WHERE db_name=?
        """,
        (db_name,),
    )
    total, tokens, wall_ms, store_ms = cur.fetchone()
    cur = stats.conn.execute(
        "SELECT COUNT(*) FROM embedding_stats WHERE db_name=? AND tokens=0 AND wall_ms=0 AND store_ms=0",
        (db_name,),
    )
    skipped = cur.fetchone()[0]
    return int(total), float(tokens), float(wall_ms), float(store_ms), int(skipped)


def _process_db(cls: Type[EmbeddableDBMixin], *, backend: str, batch_size: int) -> None:
    """Backfill embeddings for ``cls`` and print metrics."""

    stats = EmbeddingStatsDB("metrics.db")
    db_name = cls.__name__
    before = _collect_stats(stats, db_name)
    try:
        try:
            db = cls(vector_backend=backend)
        except Exception:
            db = cls()
        try:
            db.backfill_embeddings(batch_size=batch_size)  # type: ignore[call-arg]
        except TypeError:  # pragma: no cover - legacy signature
            db.backfill_embeddings()  # type: ignore[call-arg]
    except Exception as exc:  # pragma: no cover - runtime failures
        raise VectorServiceError(str(exc))
    after = _collect_stats(stats, db_name)
    delta_total = after[0] - before[0]
    delta_tokens = after[1] - before[1]
    delta_wall = after[2] - before[2]
    delta_store = after[3] - before[3]
    delta_skipped = after[4] - before[4]
    processed = delta_total - delta_skipped
    print(
        f"{db_name}: embedded {processed}, skipped {delta_skipped}, tokens {int(delta_tokens)}, "
        f"wall_ms {delta_wall:.2f}, store_ms {delta_store:.2f}"
    )


def main(
    *,
    session_id: str,
    backend: str,
    batch_size: int,
    dbs: List[str] | None,
    refresh_all: bool,
) -> None:
    """Entry point used by the CLI."""

    if refresh_all:
        seen = set()
        targets: List[str] = []
        for alias, (_, cls_name) in _DB_REGISTRY.items():
            if cls_name not in seen:
                seen.add(cls_name)
                targets.append(alias)
    else:
        if not dbs:
            raise VectorServiceError("specify --db or --all")
        targets = dbs

    total = len(targets)
    for idx, name in enumerate(targets, 1):
        cls = _resolve_db(name)
        print(f"[{idx}/{total}] backfilling {cls.__name__}")
        _process_db(cls, backend=backend, batch_size=batch_size)


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill embeddings for registered databases",
    )
    parser.add_argument(
        "--session-id",
        default="",
        help="Identifier used for metrics aggregation (unused)",
    )
    parser.add_argument(
        "--backend",
        choices=["annoy", "faiss"],
        default="annoy",
        help="Vector backend to use (default: annoy)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--db",
        action="append",
        dest="dbs",
        help="Restrict to a specific database (can be used multiple times)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Refresh every registered database",
    )
    args = parser.parse_args()
    try:
        main(
            session_id=args.session_id,
            backend=args.backend,
            batch_size=args.batch_size,
            dbs=args.dbs,
            refresh_all=args.all,
        )
    except VectorServiceError as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    cli()

