from __future__ import annotations

"""Periodic scheduler for the :mod:`metrics_aggregator`"""

import argparse
import threading
import time
from pathlib import Path
from typing import Optional

try:  # pragma: no cover - allow running as script
    from .metrics_aggregator import MetricsAggregator
except Exception:  # pragma: no cover - fallback when executed directly
    from metrics_aggregator import MetricsAggregator  # type: ignore


class MetricsScheduler:
    """Run :class:`MetricsAggregator` at regular intervals."""

    def __init__(
        self,
        db: Path | str = "metrics.db",
        out_dir: Path | str = "analytics",
        period: str = "day",
        interval: int = 3600,
    ) -> None:
        self.aggregator = MetricsAggregator(db, out_dir)
        self.period = period
        self.interval = interval
        self.running = False
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        while self.running:
            try:
                self.aggregator.run(self.period)
            except Exception:
                pass
            time.sleep(self.interval)

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=0)
            self._thread = None


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Schedule periodic metric aggregation")
    parser.add_argument("--db", default="metrics.db")
    parser.add_argument("--out-dir", default="analytics")
    parser.add_argument(
        "--period", choices=["hour", "day", "week"], default="day"
    )
    parser.add_argument(
        "--interval", type=int, help="Seconds between aggregations"
    )
    args = parser.parse_args()
    if args.interval is None:
        defaults = {"hour": 3600, "day": 86400, "week": 604800}
        args.interval = defaults[args.period]
    sched = MetricsScheduler(args.db, args.out_dir, args.period, args.interval)
    sched.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        sched.stop()


if __name__ == "__main__":
    main()
