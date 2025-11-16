from __future__ import annotations

"""Dashboard for visualising sandbox ROI history and weight changes."""

import os
import uuid

from db_router import init_db_router
from pathlib import Path
from typing import List, Callable, TYPE_CHECKING
import json

from flask import jsonify, render_template, request
if TYPE_CHECKING:  # pragma: no cover - imported for type hints only
    from flask import Request

from .metrics_dashboard import MetricsDashboard
from .roi_tracker import ROITracker
from . import synergy_weight_cli
from .alignment_dashboard import load_alignment_flag_records
from .readiness_index import military_grade_readiness

try:  # pragma: no cover - allow running as script
    from .dynamic_path_router import resolve_path  # type: ignore
except Exception:  # pragma: no cover - fallback when executed directly
    from dynamic_path_router import resolve_path  # type: ignore

MENACE_ID = uuid.uuid4().hex
LOCAL_DB_PATH = os.getenv(
    "MENACE_LOCAL_DB_PATH", str(resolve_path(f"menace_{MENACE_ID}_local.db"))
)
SHARED_DB_PATH = os.getenv(
    "MENACE_SHARED_DB_PATH", str(resolve_path("shared/global.db"))
)
GLOBAL_ROUTER = init_db_router(MENACE_ID, LOCAL_DB_PATH, SHARED_DB_PATH)


class SandboxDashboard(MetricsDashboard):
    """Serve charts for ROI history, security metrics and weight history."""

    def __init__(
        self,
        history_file: str | Path = "roi_history.json",
        weights_log: str | Path = synergy_weight_cli.LOG_PATH,
        summary_file: str | Path = resolve_path("sandbox_data") / "scenario_summary.json",
        alignment_flags_file: str | Path = resolve_path("sandbox_data") / "alignment_flags.jsonl",
        auth: Callable[[Request], bool] | None = None,
    ) -> None:
        super().__init__(history_file)
        self.load_error = ""
        self.weights_log = Path(weights_log)
        self.summary_file = Path(summary_file)
        self.alignment_flags_file = Path(alignment_flags_file)
        self._auth = auth
        if auth is not None:
            self.app.before_request(self._check_auth)
        self.app.add_url_rule('/', 'index', self.index)
        self.app.add_url_rule('/roi_data', 'roi_data', self.roi_data)
        self.app.add_url_rule('/weights', 'weights', self.weights_data)
        self.app.add_url_rule('/scenario_summary', 'scenario_summary', self.scenario_summary_data)
        self.app.add_url_rule('/relevancy', 'relevancy', self.relevancy_data)

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
    def _check_auth(self):
        if self._auth is not None and not self._auth(request):
            return "Forbidden", 403

    # ------------------------------------------------------------------
    def index(self) -> tuple[str, int]:
        self._load_tracker()
        return render_template('sandbox_dashboard.html', error=self.load_error), 200

    def roi_data(self) -> tuple[str, int]:
        tracker = self._load_tracker()
        if self.load_error:
            return jsonify({'error': self.load_error}), 500
        labels = list(range(len(tracker.roi_history)))
        security = tracker.metrics_history.get('security_score', [])
        rel_hist = tracker.metrics_history.get('synergy_reliability', [])
        safety_hist = tracker.metrics_history.get('synergy_safety_rating', [])
        res_hist = tracker.metrics_history.get('synergy_resilience', [])
        conf_hist = getattr(tracker, 'confidence_history', [])
        readiness: list[float] = []
        for i in labels:
            raroi = tracker.raroi_history[i] if i < len(tracker.raroi_history) else 0.0
            reliability = rel_hist[i] if i < len(rel_hist) else 1.0
            if i < len(conf_hist):
                reliability *= conf_hist[i]
            safety = safety_hist[i] if i < len(safety_hist) else 1.0
            resilience = res_hist[i] if i < len(res_hist) else 1.0
            readiness.append(
                military_grade_readiness(raroi, reliability, safety, resilience)
            )
        warnings = ["" for _ in labels]
        records = load_alignment_flag_records(self.alignment_flags_file)
        for idx, rec in enumerate(records):
            if idx >= len(warnings):
                break
            issues = rec.get('report', {}).get('issues', [])
            msg = "; ".join(i.get('message', '') for i in issues if i.get('message'))
            if not msg:
                sev = rec.get('severity')
                if sev is not None:
                    msg = f"severity {sev}"
            warnings[idx] = msg
        ta_meta = tracker.truth_adapter.metadata
        ta_status = {
            'needs_retrain': bool(ta_meta.get('needs_retrain', False)),
            'last_retrained': ta_meta.get('last_retrained'),
        }
        workflows = {
            wf: {
                'mae': tracker.workflow_mae(wf),
                'variance': tracker.workflow_variance(wf),
                'confidence': tracker.workflow_confidence(wf),
            }
            for wf in sorted(tracker.workflow_predicted_roi.keys())
        }
        return jsonify({
            'labels': labels,
            'roi': tracker.roi_history,
            'security': security,
            'readiness': readiness,
            'category_counts': tracker.category_summary(),
            'warnings': warnings,
            'truth_adapter': ta_status,
            'workflows': workflows,
        }), 200

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

    def scenario_summary_data(self) -> tuple[str, int]:
        try:
            with open(self.summary_file, encoding='utf-8') as fh:
                data = json.load(fh)
        except Exception as exc:
            self.logger.exception(
                "Failed to load scenario summary from %s", self.summary_file
            )
            return jsonify({'error': str(exc)}), 500
        scen_map = data.get('scenarios', {}) if isinstance(data, dict) else {}
        labels = list(scen_map.keys())
        roi = [float(scen_map[k].get('roi', 0.0)) for k in labels]
        failures = [float(scen_map[k].get('failures', 0)) for k in labels]
        successes = [float(scen_map[k].get('successes', 0)) for k in labels]
        return (
            jsonify(
                {
                    'labels': labels,
                    'roi': roi,
                    'failures': failures,
                    'successes': successes,
                }
            ),
            200,
        )

    def relevancy_data(self) -> tuple[str, int]:
        from relevancy_radar import flagged_modules

        flags = flagged_modules()
        counts: dict[str, int] = {}
        for status in flags.values():
            counts[status] = counts.get(status, 0) + 1
        return jsonify({'counts': counts}), 200


__all__ = ["SandboxDashboard"]


def cli(argv: List[str] | None = None) -> None:
    """Launch a simple HTTP dashboard for sandbox metrics."""
    import argparse

    parser = argparse.ArgumentParser(description="Sandbox dashboard")
    parser.add_argument(
        '--file',
        default=str(resolve_path('sandbox_data') / 'roi_history.json'),
        help='Path to roi_history.json'
    )
    parser.add_argument(
        '--summary-file',
        default=str(resolve_path('sandbox_data') / 'scenario_summary.json'),
        help='Path to scenario_summary.json'
    )
    parser.add_argument('--port', type=int, default=8002, help='HTTP port')
    args = parser.parse_args(argv)

    dash = SandboxDashboard(args.file, summary_file=args.summary_file)
    dash.run(port=args.port)


def main(argv: List[str] | None = None) -> None:
    cli(argv)


if __name__ == '__main__':  # pragma: no cover - CLI
    main()
