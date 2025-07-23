from __future__ import annotations

"""Simple monitoring dashboard using Flask."""

import logging
from pathlib import Path
from typing import List

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

    # ------------------------------------------------------------------
    def index(self) -> tuple[str, int]:
        msg = "Metrics dashboard running. Access /roi or /metrics for data."
        return msg, 200

    # ------------------------------------------------------------------
    def metrics(self) -> tuple[str, int]:
        metrics = {}
        for name in (
            "learning_cv_score",
            "learning_holdout_score",
            "evolution_cycle_count",
            "experiment_best_roi",
            "visual_agent_wait_time",
            "visual_agent_queue_depth",
        ):
            gauge = getattr(metrics_exporter, name, None)
            if gauge is not None:
                try:
                    metrics[name] = gauge._value.get()  # type: ignore[attr-defined]
                except Exception:
                    metrics[name] = 0.0
        return jsonify(metrics), 200

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
