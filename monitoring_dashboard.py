from __future__ import annotations

"""Simple Flask dashboard with charts for metrics, evolution history and errors."""

import threading
from typing import Iterable, Optional

from flask import Flask, jsonify, render_template_string

from .metrics_dashboard import MetricsDashboard
from .data_bot import MetricsDB
from .evolution_history_db import EvolutionHistoryDB
from .error_bot import ErrorDB
from .report_generation_bot import ReportGenerationBot, ReportOptions


_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
<h1>Menace Monitoring Dashboard</h1>
<canvas id="metrics" width="400" height="200"></canvas>
<canvas id="evolution" width="400" height="200"></canvas>
<canvas id="errors" width="400" height="200"></canvas>
<script>
async function load() {
  const m = await fetch('/metrics_data').then(r=>r.json());
  const e = await fetch('/evolution_data').then(r=>r.json());
  const err = await fetch('/error_data').then(r=>r.json());
  new Chart(document.getElementById('metrics'), {type:'line',data:{labels:m.labels,datasets:[{label:'CPU',data:m.cpu},{label:'Errors',data:m.errors}]}});
  new Chart(document.getElementById('evolution'), {type:'line',data:{labels:e.labels,datasets:[{label:'ROI',data:e.roi}]}});
  new Chart(document.getElementById('errors'), {type:'bar',data:{labels:err.labels,datasets:[{label:'Count',data:err.count}]}});
}
load();
</script>
</body>
</html>
"""


class MonitoringDashboard(MetricsDashboard):
    """Extend :class:`MetricsDashboard` with charts and reporting."""

    def __init__(
        self,
        metrics_db: MetricsDB | None = None,
        evolution_db: EvolutionHistoryDB | None = None,
        error_db: ErrorDB | None = None,
        orchestrator: object | None = None,
        history_file: str | Path = "roi_history.json",
    ) -> None:
        super().__init__(history_file)
        self.metrics_db = metrics_db or MetricsDB()
        self.evolution_db = evolution_db or EvolutionHistoryDB()
        self.error_db = error_db or ErrorDB()
        self.reporter = ReportGenerationBot(self.metrics_db)
        self.app.add_url_rule('/', 'index', self.index)
        self.app.add_url_rule('/metrics_data', 'metrics_data', self.metrics_data)
        self.app.add_url_rule('/evolution_data', 'evolution_data', self.evolution_data)
        self.app.add_url_rule('/error_data', 'error_data', self.error_data)
        self.app.add_url_rule('/status', 'status', self.status)
        self.orchestrator = orchestrator
        self._report_timer: Optional[threading.Timer] = None

    # ------------------------------------------------------------------
    def index(self) -> tuple[str, int]:
        return render_template_string(_TEMPLATE), 200

    def metrics_data(self) -> tuple[str, int]:
        rows = self.metrics_db.fetch(50)
        if hasattr(rows, 'to_dict'):
            data = rows.to_dict('records')  # type: ignore[no-any-return]
        else:
            data = rows
        labels = [r['ts'] for r in data][::-1]
        cpu = [float(r.get('cpu', 0.0)) for r in data][::-1]
        errors = [float(r.get('errors', 0.0)) for r in data][::-1]
        return jsonify({'labels': labels, 'cpu': cpu, 'errors': errors}), 200

    def evolution_data(self) -> tuple[str, int]:
        rows = self.evolution_db.fetch(50)
        labels = [r[9] for r in rows][::-1]
        roi = [float(r[3]) for r in rows][::-1]
        return jsonify({'labels': labels, 'roi': roi}), 200

    def error_data(self) -> tuple[str, int]:
        cur = self.error_db.conn.execute(
            "SELECT error_type, COUNT(*) FROM telemetry GROUP BY error_type"
        )
        rows = cur.fetchall()
        labels = [r[0] if r[0] is not None else '' for r in rows]
        count = [int(r[1]) for r in rows]
        return jsonify({'labels': labels, 'count': count}), 200

    def status(self) -> tuple[str, int]:
        if self.orchestrator and hasattr(self.orchestrator, 'status_summary'):
            try:
                data = self.orchestrator.status_summary()
                return jsonify(data), 200
            except Exception as exc:
                self.logger.warning("status_summary failed: %s", exc)
        return jsonify({}), 200

    # ------------------------------------------------------------------
    def schedule_reports(
        self,
        metrics: Iterable[str] | None = None,
        *,
        recipients: Iterable[str] | None = None,
        interval: int = 60 * 60 * 24,
    ) -> None:
        """Periodically send or save dashboard summaries."""

        options = ReportOptions(metrics=list(metrics or ['cpu', 'errors']))
        if recipients:
            options.recipients = list(recipients)
        self.reporter.schedule(options, interval=interval)

__all__ = ['MonitoringDashboard']
