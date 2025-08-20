from __future__ import annotations

"""Scheduler for periodic ranking model retraining.

This utility ties together the ranking model training utilities
with :func:`metrics_aggregator.compute_retriever_stats` so that the
ranking model and database win/regret rates stay up-to-date.  After
retraining the model it triggers a reload on any running services that
expose ``reload_ranker_model`` and ``reload_reliability_scores`` methods.
"""

from pathlib import Path
import threading
import time
from typing import Iterable, Sequence, Any, Optional
import logging

try:  # pragma: no cover - optional dependency
    from .logging_utils import log_record
except Exception:  # pragma: no cover - fallback
    try:
        from logging_utils import log_record  # type: ignore
    except Exception:  # pragma: no cover - last resort
        def log_record(**fields: object) -> dict[str, object]:  # type: ignore
            return fields

try:  # pragma: no cover - optional dependency
    from .roi_tracker import ROITracker
except Exception:  # pragma: no cover - fallback when executed directly
    from roi_tracker import ROITracker  # type: ignore

try:  # pragma: no cover - optional dependency
    from .unified_event_bus import UnifiedEventBus
except Exception:  # pragma: no cover - fallback
    try:
        from unified_event_bus import UnifiedEventBus  # type: ignore
    except Exception:  # pragma: no cover
        UnifiedEventBus = None  # type: ignore

try:  # pragma: no cover - package-relative import
    from . import retrieval_ranker as rr
    from .metrics_aggregator import compute_retriever_stats
except Exception:  # pragma: no cover - fallback when executed directly
    import retrieval_ranker as rr  # type: ignore
    from metrics_aggregator import compute_retriever_stats  # type: ignore

MODEL_PATH = Path("retrieval_ranker.model")


class RankingModelScheduler:
    """Periodically retrain the ranking model and refresh reliability KPIs."""

    def __init__(
        self,
        services: Sequence[Any],
        *,
        vector_db: Path | str = "vector_metrics.db",
        metrics_db: Path | str = "metrics.db",
        model_path: Path | str = MODEL_PATH,
        interval: int = 86400,
        roi_tracker: ROITracker | None = None,
        roi_signal_threshold: float | None = None,
        event_bus: "UnifiedEventBus" | None = None,
    ) -> None:
        self.services = list(services)
        self.vector_db = Path(vector_db)
        self.metrics_db = Path(metrics_db)
        self.model_path = Path(model_path)
        self.interval = interval
        self.roi_tracker = roi_tracker
        self.roi_signal_threshold = roi_signal_threshold
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._db_roi_counts: dict[str, int] = {}
        self.event_bus = event_bus
        if self.event_bus is None and UnifiedEventBus is not None:
            try:
                self.event_bus = UnifiedEventBus()
            except Exception:
                self.event_bus = None
        if self.event_bus is not None:
            try:
                self.event_bus.subscribe("patch_logger:outcome", self._handle_patch_outcome)
            except Exception:
                pass

    # ------------------------------------------------------------------
    def _handle_patch_outcome(self, topic: str, event: object) -> None:
        """Handle outcome notifications from :class:`PatchLogger`.

        The event is expected to include ``roi_metrics`` with per-origin ROI
        deltas.  These metrics are forwarded to :class:`ROITracker` and used to
        decide whether to trigger an immediate retrain based on
        ``roi_signal_threshold``.
        """

        metrics = {}
        if isinstance(event, dict):
            metrics = event.get("roi_metrics", {}) or {}

        if self.roi_tracker is not None and metrics:
            try:
                self.roi_tracker.update_db_metrics(metrics)
            except Exception:
                logging.exception("roi tracker update failed")

        triggered = self.roi_signal_threshold is None
        if self.roi_signal_threshold is not None:
            for stats in metrics.values():
                try:
                    if abs(float(stats.get("roi", 0.0))) >= self.roi_signal_threshold:
                        triggered = True
                        break
                except Exception:
                    continue

        if not triggered:
            return

        try:
            self.retrain_and_reload()
        except Exception:
            logging.exception("ranking model retrain failed")

    # ------------------------------------------------------------------
    def retrain_and_reload(self) -> None:
        """Retrain ranking model and notify services to reload."""
        # Update reliability KPIs before training so latest values are joined
        compute_retriever_stats(self.metrics_db)

        if self.roi_tracker is not None:
            try:
                base_roi = (
                    float(self.roi_tracker.roi_history[-1])
                    if self.roi_tracker.roi_history
                    else 0.0
                )
                _base, raroi = self.roi_tracker.calculate_raroi(
                    base_roi, workflow_type="standard", metrics={}
                )
                final_score, needs_review, confidence = self.roi_tracker.score_workflow(
                    "ranking_model", raroi
                )
                if needs_review:
                    try:
                        self.roi_tracker.borderline_bucket.add_candidate(
                            "ranking_model", float(raroi), confidence
                        )
                    except Exception:  # pragma: no cover - best effort
                        logging.exception(
                            "failed to enqueue ranking_model workflow for review"
                        )
            except Exception:
                base_roi = raroi = final_score = confidence = 0.0
                needs_review = False
            logging.info(
                "ranking retrain trigger",
                extra=log_record(
                    base_roi=base_roi,
                    raroi=raroi,
                    final_score=final_score,
                    confidence=confidence,
                    human_review=needs_review if needs_review else None,
                ),
            )

        # Train model from latest vector metrics
        df = rr.load_training_data(vector_db=self.vector_db, patch_db=self.metrics_db)
        trained = rr.train(df)
        tm = trained[0] if isinstance(trained, tuple) else trained
        rr.save_model(tm, self.model_path)

        # Reload model and reliability scores in running services
        def _reload_all(svc: Any) -> None:
            try:
                if hasattr(svc, "reload_ranker_model"):
                    svc.reload_ranker_model(self.model_path)
                if hasattr(svc, "reload_reliability_scores"):
                    svc.reload_reliability_scores()
            except Exception:
                return
            for dep in getattr(svc, "dependent_services", []) or []:
                _reload_all(dep)

        for svc in self.services:
            _reload_all(svc)

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        while self.running:
            self.retrain_and_reload()
            base_interval = self.interval
            if self.roi_tracker and self.roi_tracker.raroi_history:
                try:
                    if float(self.roi_tracker.raroi_history[-1]) < 0:
                        base_interval = max(self.interval / 2, 1)
                except Exception:
                    pass
            start = time.time()
            while self.running:
                triggered = False
                if (
                    self.roi_tracker is not None
                    and self.roi_signal_threshold is not None
                ):
                    for origin, deltas in getattr(self.roi_tracker, "origin_db_deltas", {}).items():
                        last = self._db_roi_counts.get(origin, 0)
                        if len(deltas) > last:
                            self._db_roi_counts[origin] = len(deltas)
                            if abs(deltas[-1]) >= self.roi_signal_threshold:
                                triggered = True
                    if triggered:
                        break
                if time.time() - start >= base_interval:
                    break
                time.sleep(1)

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


def main(argv: Iterable[str] | None = None) -> int:
    """CLI entry point for manual scheduling."""

    import argparse
    from importlib import import_module

    def _import_obj(path: str) -> Any:
        mod_name, attr = path.rsplit(":", 1)
        mod = import_module(mod_name)
        return getattr(mod, attr)

    p = argparse.ArgumentParser(description="Retrain ranking model periodically")
    p.add_argument("--vector-db", default="vector_metrics.db")
    p.add_argument("--metrics-db", default="metrics.db")
    p.add_argument("--model-path", default="retrieval_ranker.json")
    p.add_argument("--interval", type=int, default=86400)
    p.add_argument(
        "--service",
        action="append",
        default=[],
        help="Import path to service exposing reload_ranker_model",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    services = [_import_obj(s) for s in args.service]
    sched = RankingModelScheduler(
        services,
        vector_db=args.vector_db,
        metrics_db=args.metrics_db,
        model_path=args.model_path,
        interval=args.interval,
    )
    sched.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        sched.stop()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

