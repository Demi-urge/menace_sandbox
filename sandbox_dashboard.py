from __future__ import annotations

"""Dashboard for visualising sandbox ROI history and weight changes."""

from pathlib import Path
from typing import List
import json

from flask import jsonify, render_template_string
import logging

from .metrics_dashboard import MetricsDashboard
from .roi_tracker import ROITracker
from . import synergy_weight_cli


_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
<h1>Sandbox Dashboard</h1>
{% if error %}<p style="color:red">{{ error }}</p>{% endif %}
<canvas id="roi" width="400" height="200"></canvas>
<canvas id="security" width="400" height="200"></canvas>
<canvas id="weights" width="400" height="200"></canvas>
<script>
async function load() {
  const data = await fetch('/roi_data').then(r => r.json());
  new Chart(document.getElementById('roi'), {type:'line',data:{labels:data.labels,datasets:[{label:'ROI delta',data:data.roi}]}});
  if (data.security.length) {
    new Chart(document.getElementById('security'), {type:'line',data:{labels:data.labels.slice(0,data.security.length),datasets:[{label:'Security score',data:data.security}]}});
  }
  const wdata = await fetch('/weights').then(r => r.json());
  const ds = [];
  for (const k in wdata.weights) {
    ds.push({label:k,data:wdata.weights[k]});
  }
  if (ds.length) {
    new Chart(document.getElementById('weights'), {type:'line',data:{labels:wdata.labels,datasets:ds}});
  }
}
load();
</script>
</body>
</html>
"""


class SandboxDashboard(MetricsDashboard):
    """Serve charts for ROI history, security metrics and weight history."""

    def __init__(
        self,
        history_file: str | Path = "roi_history.json",
        weights_log: str | Path = synergy_weight_cli.LOG_PATH,
    ) -> None:
        super().__init__(history_file)
        self.load_error = ""
        self.weights_log = Path(weights_log)
        self.app.add_url_rule('/', 'index', self.index)
        self.app.add_url_rule('/roi_data', 'roi_data', self.roi_data)
        self.app.add_url_rule('/weights', 'weights', self.weights_data)

    # ------------------------------------------------------------------
    def _load_tracker(self) -> ROITracker:
        tracker = ROITracker()
        try:
            tracker.load_history(str(self.history_file))
            self.load_error = ""
        except Exception as exc:
            self.logger.exception(
                "Failed to load ROI history from %s", self.history_file
            )
            self.load_error = f"Failed to load ROI history: {exc}"
        return tracker

    # ------------------------------------------------------------------
    def index(self) -> tuple[str, int]:
        self._load_tracker()
        return render_template_string(_TEMPLATE, error=self.load_error), 200

    def roi_data(self) -> tuple[str, int]:
        tracker = self._load_tracker()
        if self.load_error:
            return jsonify({'error': self.load_error}), 500
        labels = list(range(len(tracker.roi_history)))
        security = tracker.metrics_history.get('security_score', [])
        return jsonify({'labels': labels, 'roi': tracker.roi_history, 'security': security}), 200

    def weights_data(self) -> tuple[str, int]:
        history: list[dict[str, float]] = []
        if self.weights_log.exists():
            with open(self.weights_log, encoding='utf-8') as fh:
                for line in fh:
                    try:
                        history.append(json.loads(line))
                    except Exception:
                        continue
        labels = list(range(len(history)))
        weights: dict[str, list[float]] = {}
        if history:
            keys = [k for k in history[0] if k != 'timestamp']
            for key in keys:
                weights[key] = [float(h.get(key, 0.0)) for h in history]
        return jsonify({'labels': labels, 'weights': weights}), 200


__all__ = ["SandboxDashboard"]


def cli(argv: List[str] | None = None) -> None:
    """Launch a simple HTTP dashboard for sandbox metrics."""
    import argparse

    parser = argparse.ArgumentParser(description="Sandbox dashboard")
    parser.add_argument(
        '--file',
        default=str(Path('sandbox_data') / 'roi_history.json'),
        help='Path to roi_history.json'
    )
    parser.add_argument('--port', type=int, default=8002, help='HTTP port')
    args = parser.parse_args(argv)

    dash = SandboxDashboard(args.file)
    dash.run(port=args.port)


def main(argv: List[str] | None = None) -> None:
    cli(argv)


if __name__ == '__main__':  # pragma: no cover - CLI
    main()
