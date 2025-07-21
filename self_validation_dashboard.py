from __future__ import annotations

"""Generate self-evaluation summaries for Menace."""

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from .data_bot import DataBot
from .error_forecaster import ErrorForecaster
from .dependency_update_bot import DependencyUpdater


class SelfValidationDashboard:
    """Collect metrics and write a JSON dashboard."""

    def __init__(
        self,
        data_bot: DataBot,
        error_forecaster: Optional[ErrorForecaster] = None,
        updater: Optional[DependencyUpdater] = None,
        *,
        history_file: str | Path | None = None,
    ) -> None:
        self.data_bot = data_bot
        self.error_forecaster = error_forecaster
        self.updater = updater
        self.history_file = Path(history_file) if history_file else None
        self.timer: Optional[threading.Timer] = None

    # ------------------------------------------------------------------
    def generate_report(self, path: str | Path = "dashboard.json") -> Path:
        trend = self.data_bot.long_term_roi_trend(limit=200)
        forecast = 0.0
        if self.error_forecaster:
            try:
                forecast = self.error_forecaster.forecast()
            except Exception:
                forecast = 0.0
        updates = []
        if self.updater:
            try:
                updates = self.updater._outdated()
            except Exception:
                updates = []
        report: Dict[str, Any] = {
            "roi_trend": trend,
            "error_forecast": forecast,
            "outdated": updates,
        }
        history: List[Dict[str, Any]] = []
        if self.history_file and self.history_file.exists():
            try:
                history = json.loads(self.history_file.read_text())
            except Exception:
                history = []
        if self.history_file:
            entry = {"ts": datetime.utcnow().isoformat(), "roi_trend": trend, "error_forecast": forecast}
            history.append(entry)
            self.history_file.write_text(json.dumps(history, indent=2))
            trends = [float(h.get("roi_trend", 0.0)) for h in history]
            if trends:
                report["aggregates"] = {"roi_trend_avg": sum(trends) / len(trends)}
        dest = Path(path)
        dest.write_text(json.dumps(report, indent=2))
        return dest

    # ------------------------------------------------------------------
    def schedule(self, path: str | Path = "dashboard.json", interval: int = 86400) -> None:
        """Periodically generate the dashboard at ``interval`` seconds."""

        def _loop() -> None:
            self.generate_report(path)
            self.timer = threading.Timer(interval, _loop)
            self.timer.daemon = True
            self.timer.start()

        self.timer = threading.Timer(interval, _loop)
        self.timer.daemon = True
        self.timer.start()


__all__ = ["SelfValidationDashboard", "cli", "main"]


def cli(argv: list[str] | None = None) -> None:
    """Generate periodic self validation reports."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interval",
        type=int,
        default=86400,
        help="Seconds between report generations",
    )
    parser.add_argument(
        "--output",
        default="dashboard.json",
        help="Destination JSON file",
    )
    args = parser.parse_args(argv)

    data_bot = DataBot()
    try:
        forecaster: ErrorForecaster | None = ErrorForecaster(data_bot.db)
    except Exception:  # pragma: no cover - optional deps
        forecaster = None
    try:
        updater: DependencyUpdater | None = DependencyUpdater()
    except Exception:  # pragma: no cover - optional deps
        updater = None

    dash = SelfValidationDashboard(data_bot, forecaster, updater)
    dash.generate_report(args.output)
    if args.interval > 0:
        dash.schedule(args.output, interval=args.interval)


def main(argv: list[str] | None = None) -> None:
    cli(argv)


if __name__ == "__main__":  # pragma: no cover - entry point
    main()
