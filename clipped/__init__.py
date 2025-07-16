"""Utility helpers for running various simplified Menace components.

The original module only exposed a placeholder which made integration in the
test-suite cumbersome.  This file now exposes convenience functions that allow
other modules and tests to easily invoke the scheduler and profit manager
without having to import the individual implementations manually.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from .profit_manager import ProfitManager
from .scheduler import Scheduler


def run_profit_manager(
    clips_file: str | Path,
    accounts_file: str | Path,
    topics_file: str | Path,
    chamber_file: str | Path,
) -> List[str]:
    """Run :class:`ProfitManager` with the supplied paths and return the list of
    removed clip identifiers."""

    manager = ProfitManager(Path(clips_file), Path(accounts_file), Path(topics_file), Path(chamber_file))
    return manager.run()


def build_schedule(
    clips_file: str | Path,
    topics_file: str | Path,
    accounts_file: str | Path,
    history_file: str | Path,
    topics: Iterable[str] | None = None,
) -> List[dict]:
    """Convenience wrapper around :class:`Scheduler` to compute a schedule."""

    sched = Scheduler(
        clips_file=Path(clips_file),
        topics_file=Path(topics_file),
        accounts_file=Path(accounts_file),
        history_file=Path(history_file),
    )
    sched.load()
    return sched.compute_schedule(topics)


__all__ = ["run_profit_manager", "build_schedule"]

