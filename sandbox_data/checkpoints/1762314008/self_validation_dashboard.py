from __future__ import annotations

"""Generate self-evaluation summaries for Menace."""

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from .data_bot import DataBot
from .error_forecaster import ErrorForecaster
from .dependency_update_bot import DependencyUpdater
from .error_bot import ErrorDB
from .knowledge_graph import KnowledgeGraph
from .dynamic_path_router import resolve_path


def _resolve(name: str | Path) -> Path:
    path = Path(name)
    if path.is_absolute():
        return path.resolve()
    return resolve_path(path)


class SelfValidationDashboard:
    """Collect metrics and write a JSON dashboard."""

    def __init__(
        self,
        data_bot: DataBot,
        error_forecaster: Optional[ErrorForecaster] = None,
        updater: Optional[DependencyUpdater] = None,
        graph: Optional[KnowledgeGraph] = None,
        error_db: Optional[ErrorDB] = None,
        *,
        history_file: str | Path | None = None,
    ) -> None:
        self.data_bot = data_bot
        self.error_forecaster = error_forecaster
        self.updater = updater
        self.graph = graph
        self.error_db = error_db
        self.history_file = (
            _resolve(history_file) if history_file is not None else None
        )
        self.timer: Optional[threading.Timer] = None

    # ------------------------------------------------------------------
    def generate_report(
        self, path: str | Path = resolve_path("sandbox_data") / "dashboard.json"
    ) -> Path:
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

        # Gather top error categories with affected modules from the knowledge graph
        error_stats: List[Dict[str, Any]] = []
        if self.graph and self.error_db:
            try:
                self.graph.update_error_stats(self.error_db)
                g = getattr(self.graph, "graph", None)
                if g is not None:
                    nodes = [
                        (n, d.get("weight", 0))
                        for n, d in g.nodes(data=True)
                        if n.startswith("error_type:")
                    ]
                    nodes.sort(key=lambda x: x[1], reverse=True)
                    for enode, weight in nodes[:5]:
                        modules = [
                            m.split(":", 1)[1]
                            for _, m, _ in g.out_edges(enode, data=True)
                            if m.startswith("module:")
                        ]
                        error_stats.append(
                            {
                                "error_type": enode.split(":", 1)[1],
                                "count": int(weight),
                                "modules": modules,
                            }
                        )
            except Exception:
                error_stats = []
        report["top_errors"] = error_stats
        history: List[Dict[str, Any]] = []
        if self.history_file and self.history_file.exists():
            try:
                history = json.loads(self.history_file.read_text())
            except Exception:
                history = []
        if self.history_file:
            entry = {
                "ts": datetime.utcnow().isoformat(),
                "roi_trend": trend,
                "error_forecast": forecast,
                "top_errors": error_stats,
            }
            history.append(entry)
            self.history_file.write_text(json.dumps(history, indent=2))
            trends = [float(h.get("roi_trend", 0.0)) for h in history]
            if trends:
                report["aggregates"] = {"roi_trend_avg": sum(trends) / len(trends)}
            # compile historical error trends for charting patch effectiveness
            err_trends: Dict[str, List[int]] = {}
            for h in history:
                for e in h.get("top_errors", []):
                    et = e.get("error_type")
                    if et is None:
                        continue
                    err_trends.setdefault(et, []).append(int(e.get("count", 0)))
            if err_trends:
                report["error_trends"] = err_trends
        dest = _resolve(path)
        dest.write_text(json.dumps(report, indent=2))
        return dest

    # ------------------------------------------------------------------
    def schedule(
        self,
        path: str | Path = resolve_path("sandbox_data") / "dashboard.json",
        interval: int = 86400,
    ) -> None:
        """Periodically generate the dashboard at ``interval`` seconds."""

        path = _resolve(path)

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
    try:
        graph: KnowledgeGraph | None = KnowledgeGraph()
    except Exception:  # pragma: no cover - optional deps
        graph = None
    try:
        err_db: ErrorDB | None = ErrorDB()
    except Exception:  # pragma: no cover - optional deps
        err_db = None

    dash = SelfValidationDashboard(
        data_bot,
        forecaster,
        updater,
        graph=graph,
        error_db=err_db,
    )
    dash.generate_report(args.output)
    if args.interval > 0:
        dash.schedule(args.output, interval=args.interval)


def main(argv: list[str] | None = None) -> None:
    cli(argv)


if __name__ == "__main__":  # pragma: no cover - entry point
    main()
