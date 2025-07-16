from __future__ import annotations

"""Dashboard for visualising sandbox ROI history."""

from pathlib import Path
from typing import List

from flask import jsonify, render_template_string

from .metrics_dashboard import MetricsDashboard
from .roi_tracker import ROITracker


_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
<h1>Sandbox Dashboard</h1>
<canvas id="roi" width="400" height="200"></canvas>
<canvas id="security" width="400" height="200"></canvas>
<script>
async function load() {
  const data = await fetch('/roi_data').then(r => r.json());
  new Chart(document.getElementById('roi'), {type:'line',data:{labels:data.labels,datasets:[{label:'ROI delta',data:data.roi}]}});
  if (data.security.length) {
    new Chart(document.getElementById('security'), {type:'line',data:{labels:data.labels.slice(0,data.security.length),datasets:[{label:'Security score',data:data.security}]}});
  }
}
load();
</script>
</body>
</html>
"""


class SandboxDashboard(MetricsDashboard):
    """Serve charts for ROI history and security metrics."""

    def __init__(self, history_file: str | Path = "roi_history.json") -> None:
        super().__init__(history_file)
        self.app.add_url_rule('/', 'index', self.index)
        self.app.add_url_rule('/roi_data', 'roi_data', self.roi_data)

    # ------------------------------------------------------------------
    def _load_tracker(self) -> ROITracker:
        tracker = ROITracker()
        try:
            tracker.load_history(str(self.history_file))
        except Exception:
            pass
        return tracker

    # ------------------------------------------------------------------
    def index(self) -> tuple[str, int]:
        return render_template_string(_TEMPLATE), 200

    def roi_data(self) -> tuple[str, int]:
        tracker = self._load_tracker()
        labels = list(range(len(tracker.roi_history)))
        security = tracker.metrics_history.get('security_score', [])
        return jsonify({'labels': labels, 'roi': tracker.roi_history, 'security': security}), 200


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
