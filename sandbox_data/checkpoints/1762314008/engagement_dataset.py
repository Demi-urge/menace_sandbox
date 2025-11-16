import csv
from typing import Optional, Tuple
import os

from .sql_db import create_session, RLFeedback, ensure_schema


def _features(text: str) -> Tuple[int, int, int]:
    """Return length, exclamation count and question mark count for text."""
    return len(text), text.count("!"), text.count("?")


def collect_engagement_logs(
    path: str,
    *,
    session_factory: Optional[callable] = None,
    db_url: Optional[str] = None,
) -> str:
    """Export engagement training data to ``path``.

    The dataset contains ``length``, ``exclam``, ``question`` and ``engagement``
    columns derived from :class:`RLFeedback` records.
    """
    if session_factory is None:
        ensure_schema(db_url or os.environ.get("NEURO_DB_URL", "sqlite://"))
        session_factory = create_session(db_url)

    Session = session_factory
    with Session() as s:
        rows = s.query(RLFeedback).order_by(RLFeedback.id).all()

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["length", "exclam", "question", "engagement"])
        for r in rows:
            length, exclam, question = _features(r.text)
            writer.writerow([length, exclam, question, float(r.score)])

    return path


__all__ = ["collect_engagement_logs"]
