from __future__ import annotations

"""Simple monitoring dashboard using Flask."""

import logging
from pathlib import Path
from typing import List, Dict, TYPE_CHECKING
import json
from io import BytesIO
import queue
import statistics
from collections import defaultdict
import os
import uuid

from flask import Flask, jsonify, send_file, request

import sys
from types import ModuleType

from .telemetry_backend import TelemetryBackend, get_table_access_counts

if TYPE_CHECKING:  # pragma: no cover - used for type checking only
    from .roi_tracker import ROITracker

try:  # pragma: no cover - allow running as script
    from .db_router import init_db_router  # type: ignore
except Exception:  # pragma: no cover - fallback when executed directly
    from db_router import init_db_router  # type: ignore

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

router = GLOBAL_ROUTER


def _get_metrics_exporter() -> ModuleType | None:
    """Return the preloaded ``metrics_exporter`` module if available."""

    pkg = sys.modules.get(__package__ or "")
    me = getattr(pkg, "metrics_exporter", None)
    if me is not None:
        return me
    try:  # pragma: no cover - executed only when not preloaded
        from . import metrics_exporter as me  # type: ignore
    except Exception:
        return None
    return me


class MetricsDashboard:
    """Expose basic metrics and ROI history for dashboards."""

    def __init__(self, history_file: str | Path = "roi_history.json") -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.app = Flask(__name__)
        self.history_file = Path(history_file)
        self.app.add_url_rule("/", "index", self.index)
        self.app.add_url_rule("/dashboard", "dashboard", self.index)
        self.app.add_url_rule("/metrics", "metrics", self.metrics)
        self.app.add_url_rule("/roi", "roi", self.roi)
        self.app.add_url_rule("/readiness", "readiness", self.readiness)
        self.app.add_url_rule("/metrics/<name>", "metric_series", self.metric_series)
        self.app.add_url_rule(
            "/plots/predictions.png", "plot_predictions", self.plot_predictions
        )
        self.app.add_url_rule(
            "/plots/prediction_error.png",
            "plot_prediction_error",
            self.plot_prediction_error,
        )
        self.app.add_url_rule(
            "/plots/roi_category_distribution.png",
            "plot_roi_categories",
            self.plot_roi_categories,
        )
        self.app.add_url_rule(
            "/plots/roi_improvement.png",
            "plot_roi_improvement",
            self.plot_roi_improvement,
        )
        self.app.add_url_rule(
            "/plots/prediction_stats.png",
            "plot_prediction_stats",
            self.plot_prediction_stats,
        )
        self.app.add_url_rule(
            "/plots/readiness_distribution.png",
            "plot_readiness_distribution",
            self.plot_readiness_distribution,
        )
        self.app.add_url_rule(
            "/vector_heatmap", "vector_heatmap_default", self.vector_heatmap
        )
        self.app.add_url_rule(
            "/vector_heatmap/<period>", "vector_heatmap", self.vector_heatmap
        )
        self.app.add_url_rule(
            "/aggregates/<metric>/<period>", "aggregates", self.aggregates
        )
        self.app.add_url_rule(
            "/heatmaps/<metric>/<period>.png", "agg_heatmap", self.agg_heatmap
        )
        self._refresh_queue: queue.Queue[str] = queue.Queue()
        self.app.add_url_rule("/refresh", "refresh", self.refresh)
        self.app.add_url_rule(
            "/refresh/stream", "refresh_stream", self.refresh_stream
        )
        self.app.add_url_rule(
            "/scenario_deltas/<workflow_id>",
            "scenario_deltas_chart",
            self.scenario_deltas_chart,
        )

    # ------------------------------------------------------------------
    def index(self) -> tuple[str, int]:
        msg = "Metrics dashboard running. Access /roi or /metrics for data."
        return msg, 200

    # ------------------------------------------------------------------
    def metrics(self) -> tuple[str, int]:
        metrics = {}

        exporter = _get_metrics_exporter()

        def _get_value(gauge: object) -> float:
            try:
                return gauge._value.get()  # type: ignore[attr-defined]
            except Exception:
                try:
                    wrappers = getattr(gauge, "_values", None)
                    if wrappers:
                        return float(sum(w.get() for w in wrappers.values()))
                    wrappers = getattr(gauge, "_metrics", None)
                    if wrappers:
                        return float(
                            sum(getattr(w, "_value", 0).get() for w in wrappers.values())
                        )
                except Exception:
                    return 0.0

        names = (
            "learning_cv_score",
            "learning_holdout_score",
            "evolution_cycle_count",
            "experiment_best_roi",
            "container_creation_success_total",
            "container_creation_failures_total",
            "container_creation_alerts_total",
            "synergy_weight_update_failures_total",
            "retired_modules_total",
            "compressed_modules_total",
            "replaced_modules_total",
            "sandbox_cpu_percent",
            "sandbox_memory_mb",
            "sandbox_crashes_total",
        )
        if exporter is not None:
            for name in names:
                gauge = getattr(exporter, name, None)
                if gauge is not None:
                    metrics[name] = _get_value(gauge)

            # Include per-status counts and impact totals from relevancy radar
            flag_gauge = getattr(exporter, "relevancy_flagged_modules_total", None)
            if flag_gauge is not None:
                try:
                    metrics["relevancy_flagged_modules_total"] = {
                        k[0]: v.get() for k, v in getattr(flag_gauge, "_values", {}).items()
                    }
                except Exception:
                    metrics["relevancy_flagged_modules_total"] = {}

            impact_gauge = getattr(exporter, "relevancy_flagged_modules_impact_total", None)
            if impact_gauge is not None:
                try:
                    metrics["relevancy_flagged_modules_impact_total"] = {
                        k[0]: v.get() for k, v in getattr(impact_gauge, "_values", {}).items()
                    }
                except Exception:
                    metrics["relevancy_flagged_modules_impact_total"] = {}

            for name in ("workflow_confidence", "workflow_mae", "workflow_variance"):
                gauge = getattr(exporter, name, None)
                if gauge is not None:
                    try:
                        metrics[name] = {
                            k[0]: v.get() for k, v in getattr(gauge, "_values", {}).items()
                        }
                    except Exception:
                        metrics[name] = {}

        try:
            from . import synergy_auto_trainer as sat

            for name in (
                "synergy_trainer_iterations",
                "synergy_trainer_failures_total",
            ):
                gauge = getattr(sat, name, None)
                if gauge is not None:
                    metrics[name] = _get_value(gauge)
        except Exception:
            pass

        try:
            with router.get_connection("retriever_kpi") as conn:
                rows = conn.execute(
                    """
            SELECT origin_db, roi FROM retriever_kpi
            WHERE (origin_db, ts) IN (
                SELECT origin_db, max(ts) FROM retriever_kpi GROUP BY origin_db
            )
            ORDER BY roi DESC
            """
                ).fetchall()
            metrics["roi_rankings"] = rows
        except Exception:
            pass

        # Include DBRouter table usage metrics
        try:
            metrics["table_access"] = get_table_access_counts()
        except Exception:
            metrics["table_access"] = {}

        # Highlight modules recently flagged by relevancy radar with impact scores
        try:
            from .relevancy_radar import flagged_modules as rr_flagged

            flags = rr_flagged()
            impacts: dict[str, float] = {}
            metrics_path = Path(resolve_path("sandbox_data/relevancy_metrics.json"))
            try:
                data = json.loads(metrics_path.read_text())
                if isinstance(data, dict):
                    for mod, info in data.items():
                        if isinstance(info, dict):
                            impacts[str(mod)] = float(info.get("impact", 0.0))
            except Exception:
                pass
            if flags:
                metrics["relevancy_flags"] = [
                    {
                        "module": m,
                        "status": s,
                        "impact": impacts.get(m, 0.0),
                    }
                    for m, s in sorted(flags.items())
                ]
        except Exception:
            pass

        return jsonify(metrics), 200

    # ------------------------------------------------------------------
    def vector_heatmap(self, period: str = "hourly") -> tuple[str, int]:
        """Return aggregated vector metrics suitable for heatmaps."""
        path = Path(f"vector_metrics_heatmap_{period}.json")
        if not path.exists():
            return jsonify([]), 200
        try:
            data = json.loads(path.read_text())
        except Exception:
            data = []
        return jsonify(data), 200

    def refresh(self) -> tuple[str, int]:
        """Pull latest telemetry and readiness stats and notify listeners."""
        try:
            tb = TelemetryBackend()
            scope = request.args.get("scope") or "local"
            history = tb.fetch_history(scope=scope)
        except Exception:
            history = []
        readiness: dict[str, float] = {}
        for h in history:
            wf = h.get("workflow_id")
            ready = h.get("readiness")
            if wf is not None and ready is not None:
                readiness[str(wf)] = float(ready)
        self._refresh_queue.put("refresh")
        return jsonify({"telemetry": history, "readiness": readiness}), 200

    def refresh_stream(self):
        def _gen():
            while True:
                self._refresh_queue.get()
                yield "data: refresh\n\n"

        return self.app.response_class(_gen(), mimetype="text/event-stream")

    # ------------------------------------------------------------------
    def aggregates(self, metric: str, period: str) -> tuple[str, int]:
        """Expose aggregated metrics from :mod:`metrics_aggregator`."""

        allowed_metrics = {"embedding_stats", "retrieval_stats", "retriever_kpi"}
        allowed_periods = {"hour", "day", "week"}
        if metric not in allowed_metrics or period not in allowed_periods:
            return jsonify([]), 200
        table = f"{metric}_agg_{period}"
        rows: List[dict] = []
        try:
            with router.get_connection(table) as conn:
                cur = conn.execute(f"SELECT * FROM {table}")
                cols = [c[0] for c in cur.description]
                rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        except Exception:
            # If aggregation hasn't been run yet, attempt to generate it on
            # demand.  Failures are silently ignored so this dashboard remains
            # lightweight in constrained environments.
            try:
                from .metrics_aggregator import MetricsAggregator

                MetricsAggregator("metrics.db", "analytics").run(period)
                with router.get_connection(table) as conn:
                    cur = conn.execute(f"SELECT * FROM {table}")
                    cols = [c[0] for c in cur.description]
                    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            except Exception:
                pass
        return jsonify(rows), 200

    # ------------------------------------------------------------------
    def agg_heatmap(self, metric: str, period: str):
        """Serve heatmap PNGs generated by the aggregator."""

        path = Path(resolve_path("analytics/heatmaps")) / f"{metric}_{period}.png"
        if not path.exists():
            try:
                from .metrics_aggregator import MetricsAggregator

                MetricsAggregator("metrics.db", "analytics").run(period)
            except Exception:
                pass
        if not path.exists():
            return ("", 404)
        return send_file(str(path), mimetype="image/png")

    # ------------------------------------------------------------------
    def _load_tracker(self) -> "ROITracker":
        from .roi_tracker import ROITracker

        tracker = ROITracker()
        try:
            tracker.load_history(str(self.history_file))
        except Exception:
            pass
        return tracker

    def roi(self) -> tuple[str, int]:
        tracker = self._load_tracker()
        labels: List[int] = list(range(len(tracker.roi_history)))
        data = {
            "labels": labels,
            "roi": tracker.roi_history,
            "predicted": tracker.predicted_roi,
            "actual": tracker.actual_roi,
            "category_counts": tracker.category_summary(),
        }
        # Visualise the penalty applied by risk adjustments over time as the
        # gap between raw ROI and risk-adjusted ROI (RAROI).
        if tracker.raroi_history:
            data["risk_penalty"] = [
                r - rr for r, rr in zip(tracker.roi_history, tracker.raroi_history)
            ]
        # Expose per-database ROI deltas so dashboards can chart contribution
        # trends for each origin database over time.
        try:
            data["origin_db_deltas"] = {
                db: list(vals)
                for db, vals in tracker.origin_db_delta_history.items()
            }
        except Exception:
            data["origin_db_deltas"] = {}
        # Include latest per-database contribution and outcome rates.
        try:
            data["db_metrics"] = tracker.db_roi_report()
        except Exception:
            data["db_metrics"] = []
        data["workflow_mae"] = {
            k: v for k, v in tracker.workflow_mae_history.items() if k != "_global"
        }
        data["workflow_variance"] = {
            k: v
            for k, v in tracker.workflow_variance_history.items()
            if k != "_global"
        }
        data["workflow_confidence"] = {
            k: v
            for k, v in tracker.workflow_confidence_history.items()
            if k != "_global"
        }
        synergy_names = {
            n for n in tracker.metrics_history if n.startswith("synergy_")
        } | set(tracker.synergy_metrics_history)
        for m in sorted(synergy_names):
            s_hist = tracker.synergy_metrics_history.get(m)
            if s_hist is None:
                s_hist = tracker.metrics_history.get(m, [])
            data[m] = s_hist
            data[f"{m}_predicted"] = tracker.predicted_metrics.get(m, [])
            data[f"{m}_actual"] = tracker.actual_metrics.get(m, [])
        return jsonify(data), 200

    def metric_series(self, name: str) -> tuple[str, int]:
        tracker = self._load_tracker()
        hist = tracker.metrics_history.get(name)
        if hist is None:
            hist = tracker.synergy_metrics_history.get(name, [])
        preds = tracker.predicted_metrics.get(name, [])
        acts = tracker.actual_metrics.get(name, [])

        syn_name = f"synergy_{name}" if not name.startswith("synergy_") else None
        syn_hist = []
        syn_pred = []
        syn_act = []
        if syn_name:
            syn_hist = tracker.metrics_history.get(syn_name)
            if syn_hist is None:
                syn_hist = tracker.synergy_metrics_history.get(syn_name, [])
            syn_pred = tracker.predicted_metrics.get(syn_name, [])
            syn_act = tracker.actual_metrics.get(syn_name, [])

        length = max(
            len(hist), len(preds), len(acts), len(syn_hist), len(syn_pred), len(syn_act)
        )
        labels: List[int] = list(range(length))

        data = {
            "labels": labels,
            "values": hist,
            "predicted": preds,
            "actual": acts,
        }
        if syn_name and (syn_hist or syn_pred or syn_act):
            data["synergy_values"] = syn_hist
            data["synergy_predicted"] = syn_pred
            data["synergy_actual"] = syn_act

        return jsonify(data), 200

    def readiness(self) -> tuple[str, int]:
        """Render readiness overview with Chart.js plots."""
        history: list[dict] = []
        try:
            tb = TelemetryBackend()
            scope = request.args.get("scope") or "local"
            history = tb.fetch_history(scope=scope)
        except Exception:
            history = []

        labels = sorted({h.get("ts") for h in history})
        readiness_map: dict[str, dict[str, float | None]] = {}
        error_map: dict[str, dict[str, tuple[float, bool]]] = {}
        readiness_values: list[float] = []
        flagged: set[str] = set()

        for h in history:
            wf = h.get("workflow_id") or "unknown"
            ts = h.get("ts")
            ready = h.get("readiness")
            readiness_map.setdefault(wf, {})[ts] = ready
            if ready is not None:
                readiness_values.append(ready)
            pred = h.get("predicted")
            act = h.get("actual")
            if pred is not None and act is not None:
                err = abs(pred - act)
                error_map.setdefault(wf, {})[ts] = (err, bool(h.get("drift_flag")))
            if (ready is not None and ready < 0.5) or h.get("drift_flag"):
                flagged.add(wf)

        ready_datasets = []
        for wf, ts_map in readiness_map.items():
            data = [ts_map.get(ts) for ts in labels]
            ready_datasets.append({"label": wf, "data": data, "fill": False})

        error_datasets = []
        for wf, ts_map in error_map.items():
            data = [ts_map.get(ts, (None, False))[0] for ts in labels]
            colors = [
                "red" if ts_map.get(ts, (0.0, False))[1] else "blue" for ts in labels
            ]
            error_datasets.append(
                {
                    "label": wf,
                    "data": data,
                    "fill": False,
                    "pointBackgroundColor": colors,
                }
            )

        bins = [0] * 10
        for r in readiness_values:
            idx = min(int(r * 10), 9)
            bins[idx] += 1
        bin_labels = [f"{i/10:.1f}-{(i+1)/10:.1f}" for i in range(10)]

        flagged_html = (
            "".join(f"<li>{wf}</li>" for wf in sorted(flagged)) if flagged else "<li>None</li>"
        )

        html = f"""
<html>
<head>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
  <h1>Readiness Dashboard</h1>
  <div><canvas id="readiness_over_time"></canvas></div>
  <div><canvas id="readiness_hist"></canvas></div>
  <div><canvas id="error_rate_chart"></canvas></div>
  <h2>Flagged Instability</h2>
  <ul>{flagged_html}</ul>
  <script>
    const labels = {json.dumps(labels)};
    new Chart(document.getElementById('readiness_over_time'), {{
      type: 'line',
      data: {{ labels: labels, datasets: {json.dumps(ready_datasets)} }}
    }});
    new Chart(document.getElementById('readiness_hist'), {{
      type: 'bar',
      data: {{
        labels: {json.dumps(bin_labels)},
        datasets: [{{ label: 'Readiness distribution', data: {json.dumps(bins)} }}]
      }}
    }});
    new Chart(document.getElementById('error_rate_chart'), {{
      type: 'line',
      data: {{ labels: labels, datasets: {json.dumps(error_datasets)} }}
    }});
  </script>
</body>
</html>
        """
        return html, 200

    # ------------------------------------------------------------------
    def scenario_deltas_chart(self, workflow_id: str) -> tuple[str, int]:
        """Return ROI/RAROI deltas per scenario with trends and confidence."""

        tb = TelemetryBackend()
        scenario_name = request.args.get("scenario")
        start = request.args.get("start")
        end = request.args.get("end")
        scope = request.args.get("scope") or "local"
        history = tb.fetch_history(
            workflow_id,
            scenario=scenario_name,
            start_ts=start,
            end_ts=end,
            scope=scope,
        )

        labels: List[str] = []
        scenarios: Dict[str, Dict[str, List[float]]] = {}
        for h in history:
            labels.append(h.get("ts"))
            deltas = h.get("scenario_deltas", {})
            for name, delta in deltas.items():
                d = scenarios.setdefault(
                    name,
                    {"roi": [], "raroi": [], "trend": [], "ci_low": [], "ci_high": []},
                )
                try:
                    val = float(delta)
                except Exception:
                    val = 0.0
                d["roi"].append(val)
                d["raroi"].append(val)

        for data in scenarios.values():
            roi_vals = data["roi"]
            if not roi_vals:
                continue
            cumulative: List[float] = []
            total = 0.0
            for i, v in enumerate(roi_vals, 1):
                total += v
                cumulative.append(total / i)
            data["trend"] = cumulative
            if len(roi_vals) > 1:
                mean_val = statistics.mean(roi_vals)
                try:
                    stdev = statistics.stdev(roi_vals)
                except statistics.StatisticsError:
                    stdev = 0.0
                margin = 1.96 * stdev / (len(roi_vals) ** 0.5)
                low = mean_val - margin
                high = mean_val + margin
                data["ci_low"] = [low] * len(roi_vals)
                data["ci_high"] = [high] * len(roi_vals)
            else:
                data["ci_low"] = roi_vals[:]
                data["ci_high"] = roi_vals[:]

        return jsonify({"labels": labels, "scenarios": scenarios}), 200

    def plot_predictions(self) -> tuple[bytes, int, dict[str, str]]:
        """Return a PNG plot showing predicted vs. actual values."""
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:
            return b"", 200, {"Content-Type": "image/png"}

        tracker = self._load_tracker()

        base_metrics = sorted(
            m for m in tracker.metrics_history if not m.startswith("synergy_")
        )
        synergy_extra = [
            m
            for m in tracker.metrics_history
            if m.startswith("synergy_")
            and (
                m[len("synergy_"):] not in base_metrics
                or m
                in {
                    "synergy_profitability",
                    "synergy_revenue",
                    "synergy_projected_lucrativity",
                    "synergy_maintainability",
                    "synergy_code_quality",
                    "synergy_network_latency",
                    "synergy_throughput",
                }
            )
            and m != "synergy_roi"
        ]

        n = len(base_metrics) + len(synergy_extra) + 1
        fig, axes = plt.subplots(n, 1, figsize=(6, 3 * n))

        def _plot(
            ax,
            label: str,
            pred: list[float],
            act: list[float],
            syn_pred: list[float] | None = None,
            syn_act: list[float] | None = None,
        ) -> None:
            ax.plot(pred, label="predicted")
            ax.plot(act, label="actual")
            if syn_pred or syn_act:
                if syn_pred:
                    ax.plot(syn_pred, label="synergy_predicted")
                if syn_act:
                    ax.plot(syn_act, label="synergy_actual")
            ax.set_title(label)
            ax.legend()

        _plot(
            axes[0] if n > 1 else axes,
            "ROI",
            tracker.predicted_roi,
            tracker.actual_roi,
            tracker.predicted_metrics.get("synergy_roi"),
            tracker.actual_metrics.get("synergy_roi"),
        )

        idx = 1
        for name in base_metrics:
            _plot(
                axes[idx] if n > 1 else axes,
                name,
                tracker.predicted_metrics.get(name, []),
                tracker.actual_metrics.get(name, []),
                tracker.predicted_metrics.get(f"synergy_{name}"),
                tracker.actual_metrics.get(f"synergy_{name}"),
            )
            idx += 1

        for name in synergy_extra:
            _plot(
                axes[idx] if n > 1 else axes,
                name,
                tracker.predicted_metrics.get(name, []),
                tracker.actual_metrics.get(name, []),
            )
            idx += 1

        fig.tight_layout()
        from io import BytesIO

        buf = BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue(), 200, {"Content-Type": "image/png"}

    def plot_readiness_distribution(self) -> tuple[bytes, int, dict[str, str]]:
        """Return a PNG histogram of readiness scores."""
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:
            return b"", 200, {"Content-Type": "image/png"}

        try:
            tb = TelemetryBackend()
            scope = request.args.get("scope") or "local"
            history = tb.fetch_history(scope=scope)
            values = [h["readiness"] for h in history if h.get("readiness") is not None]
        except Exception:
            values = []

        fig, ax = plt.subplots()
        if values:
            ax.hist(values, bins=10, range=(0, 1))
        ax.set_title("Readiness Distribution")
        ax.set_xlabel("Readiness")
        ax.set_ylabel("Count")

        buf = BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue(), 200, {"Content-Type": "image/png"}

    def plot_prediction_error(self) -> tuple[bytes, int, dict[str, str]]:
        """Return a PNG plot of prediction error and confidence statistics."""
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:
            return b"", 200, {"Content-Type": "image/png"}

        tracker = self._load_tracker()
        errors = [abs(p - a) for p, a in zip(tracker.predicted_roi, tracker.actual_roi)]
        confidences = list(tracker.confidence_history)
        drift_flags = list(tracker.drift_flags)

        fig = plt.figure(figsize=(10, 8))
        gs = fig.add_gridspec(3, 2)
        ax_err = fig.add_subplot(gs[0, :])
        ax_horizon = fig.add_subplot(gs[1, :])
        ax_conf = fig.add_subplot(gs[2, 0])
        ax_conf_dist = fig.add_subplot(gs[2, 1])

        # Plot raw prediction errors with outlier annotation
        ax_err.plot(errors, label="error")
        if len(errors) >= 2:
            mean_err = statistics.mean(errors)
            stdev_err = statistics.stdev(errors)
            thresh = mean_err + 2 * stdev_err
            outliers = [i for i, e in enumerate(errors) if e > thresh]
            if outliers:
                ax_err.scatter(
                    outliers,
                    [errors[i] for i in outliers],
                    color="red",
                    label="outlier",
                )
                for i in outliers:
                    ax_err.annotate(f"{errors[i]:.2f}", (i, errors[i]))
        ax_err.set_title("Prediction Error Over Time")
        ax_err.set_xlabel("Iteration")
        ax_err.set_ylabel("Absolute Error")
        ax_err.legend()

        # Error distribution per horizon
        horizon_errs: dict[int, list[float]] = defaultdict(list)
        for hist in tracker.horizon_mae_history:
            for h, v in hist.items():
                horizon_errs[int(h)].append(float(v))
        if horizon_errs:
            horizons = sorted(horizon_errs)
            data = [horizon_errs[h] for h in horizons]
            ax_horizon.boxplot(data, labels=horizons)
            ax_horizon.set_ylabel("MAE")
        else:
            ax_horizon.text(0.5, 0.5, "No horizon data", ha="center", va="center")
        ax_horizon.set_title("Error Distribution by Horizon")
        ax_horizon.set_xlabel("Horizon")

        # Confidence evolution with drift and decay highlighting
        ax_conf.plot(confidences, label="confidence")
        for i, flag in enumerate(drift_flags[: len(confidences)]):
            if flag:
                ax_conf.axvspan(
                    i - 0.5,
                    i + 0.5,
                    color="orange",
                    alpha=0.3,
                    label="drift" if i == 0 else "",
                )
        decay_start = None
        for i in range(1, len(confidences)):
            if confidences[i] < confidences[i - 1]:
                if decay_start is None:
                    decay_start = i - 1
            elif decay_start is not None:
                ax_conf.axvspan(
                    decay_start,
                    i - 1,
                    color="red",
                    alpha=0.1,
                    label="decay" if decay_start == 0 else "",
                )
                decay_start = None
        if decay_start is not None and len(confidences) - 1 > decay_start:
            ax_conf.axvspan(
                decay_start,
                len(confidences) - 1,
                color="red",
                alpha=0.1,
                label="decay" if decay_start == 0 else "",
            )
        if len(confidences) >= 2:
            mean_conf = statistics.mean(confidences)
            stdev_conf = statistics.stdev(confidences)
            low_conf = [
                i for i, c in enumerate(confidences) if c < mean_conf - 2 * stdev_conf
            ]
            if low_conf:
                ax_conf.scatter(
                    low_conf,
                    [confidences[i] for i in low_conf],
                    color="red",
                    marker="x",
                    label="outlier",
                )
        ax_conf.set_title("Confidence Scores Over Time")
        ax_conf.set_xlabel("Iteration")
        ax_conf.set_ylabel("Confidence")
        ax_conf.legend()

        # Distribution of confidence scores
        if confidences:
            bins = min(10, max(1, len(confidences) // 2))
            ax_conf_dist.hist(confidences, bins=bins, color="gray", alpha=0.7)
        ax_conf_dist.set_title("Confidence Score Distribution")
        ax_conf_dist.set_xlabel("Confidence")
        ax_conf_dist.set_ylabel("Frequency")

        fig.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue(), 200, {"Content-Type": "image/png"}

    def plot_roi_categories(self) -> tuple[bytes, int, dict[str, str]]:
        """Return a PNG bar chart of predicted ROI category distribution."""
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:
            return b"", 200, {"Content-Type": "image/png"}

        tracker = self._load_tracker()
        counts = tracker.category_summary()
        if not counts:
            return b"", 200, {"Content-Type": "image/png"}
        labels = list(counts.keys())
        values = list(counts.values())
        fig, ax = plt.subplots()
        ax.bar(range(len(labels)), values)
        ax.set_title("Predicted ROI Category Distribution")
        ax.set_ylabel("Count")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        from io import BytesIO

        buf = BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue(), 200, {"Content-Type": "image/png"}

    def plot_roi_improvement(self) -> tuple[bytes, int, dict[str, str]]:
        """Return a PNG plot of predicted vs. actual ROI improvements."""
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:
            return b"", 200, {"Content-Type": "image/png"}

        tracker = self._load_tracker()
        fig, ax = plt.subplots()
        ax.plot(tracker.predicted_roi, label="predicted")
        ax.plot(tracker.actual_roi, label="actual")
        ax.set_title("Actual ROI Improvements vs Predictions")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("ROI")
        ax.legend()
        from io import BytesIO

        buf = BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue(), 200, {"Content-Type": "image/png"}

    def plot_prediction_stats(self) -> tuple[bytes, int, dict[str, str]]:
        """Return a PNG plot of MAE and classification accuracy trends."""
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:
            return b"", 200, {"Content-Type": "image/png"}

        tracker = self._load_tracker()
        mae_trend = tracker.rolling_mae_trend()  # use full history
        acc_trend = tracker.rolling_accuracy_trend()
        if not mae_trend and not acc_trend:
            return b"", 200, {"Content-Type": "image/png"}

        fig, ax1 = plt.subplots()
        ax1.plot(mae_trend, label="MAE", color="tab:blue")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("MAE", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax2 = ax1.twinx()
        ax2.plot(acc_trend, label="Accuracy", color="tab:green")
        ax2.set_ylabel("Accuracy", color="tab:green")
        ax2.tick_params(axis="y", labelcolor="tab:green")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="best")
        fig.tight_layout()
        from io import BytesIO

        buf = BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue(), 200, {"Content-Type": "image/png"}

    def run(self, host: str = "0.0.0.0", port: int = 5000) -> None:
        self.app.run(host=host, port=port)


__all__ = ["MetricsDashboard"]


def cli(argv: List[str] | None = None) -> None:
    """Launch a metrics dashboard HTTP server."""
    import argparse

    parser = argparse.ArgumentParser(description="Metrics dashboard")
    parser.add_argument(
        "--file",
        default="roi_history.json",
        help="Path to roi_history.json",
    )
    parser.add_argument("--port", type=int, default=5000, help="HTTP port")
    args = parser.parse_args(argv)

    dash = MetricsDashboard(args.file)
    dash.run(port=args.port)


def main(argv: List[str] | None = None) -> None:
    cli(argv)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
