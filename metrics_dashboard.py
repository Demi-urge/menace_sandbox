from __future__ import annotations

"""Simple monitoring dashboard using Flask."""

import logging
import sqlite3
from pathlib import Path
from typing import List
import json

from flask import Flask, jsonify

from . import metrics_exporter


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
            "/vector_heatmap", "vector_heatmap", self.vector_heatmap
        )

    # ------------------------------------------------------------------
    def index(self) -> tuple[str, int]:
        msg = "Metrics dashboard running. Access /roi or /metrics for data."
        return msg, 200

    # ------------------------------------------------------------------
    def metrics(self) -> tuple[str, int]:
        metrics = {}

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
            "visual_agent_wait_time",
            "visual_agent_queue_depth",
            "visual_agent_recoveries_total",
            "container_creation_success_total",
            "container_creation_failures_total",
            "container_creation_alerts_total",
            "synergy_weight_update_failures_total",
        )
        for name in names:
            gauge = getattr(metrics_exporter, name, None)
            if gauge is not None:
                metrics[name] = _get_value(gauge)

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
            with sqlite3.connect("metrics.db") as conn:
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

        return jsonify(metrics), 200

    # ------------------------------------------------------------------
    def vector_heatmap(self) -> tuple[str, int]:
        """Return aggregated vector metrics suitable for heatmaps."""
        path = Path("vector_metrics_heatmap.json")
        if not path.exists():
            return jsonify([]), 200
        try:
            data = json.loads(path.read_text())
        except Exception:
            data = []
        return jsonify(data), 200

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
                m[len("synergy_") :] not in base_metrics
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

    def plot_prediction_error(self) -> tuple[bytes, int, dict[str, str]]:
        """Return a PNG plot of absolute prediction error over time."""
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:
            return b"", 200, {"Content-Type": "image/png"}

        tracker = self._load_tracker()
        errors = [abs(p - a) for p, a in zip(tracker.predicted_roi, tracker.actual_roi)]
        fig, ax = plt.subplots()
        ax.plot(errors, label="error")
        ax.set_title("Prediction Error Over Time")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Absolute Error")
        ax.legend()
        from io import BytesIO

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
