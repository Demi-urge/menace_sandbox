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
import pandas as pd
import logging
import json
import os

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
    from .retrieval_training_dataset import build_dataset
except Exception:  # pragma: no cover - fallback when executed directly
    import retrieval_ranker as rr  # type: ignore
    from metrics_aggregator import compute_retriever_stats  # type: ignore
    from retrieval_training_dataset import build_dataset  # type: ignore

try:  # pragma: no cover - optional dependency
    from .vector_metrics_db import VectorMetricsDB
except Exception:  # pragma: no cover - fallback when executed directly
    from vector_metrics_db import VectorMetricsDB  # type: ignore


def needs_retrain(vector_db: Path | str, win_rate_threshold: float = 0.5) -> bool:
    """Return ``True`` if retraining should occur based on win rate.

    The helper inspects :class:`VectorMetricsDB` and compares the overall
    ``retriever_win_rate`` against ``win_rate_threshold``.  A lower win rate
    suggests the ranking model has drifted and should be retrained.
    ``vector_db`` may be a :class:`~pathlib.Path` or path string pointing to the
    SQLite database file.
    """

    try:
        db = VectorMetricsDB(vector_db)
        try:
            win_rate = db.retriever_win_rate()
        finally:
            try:
                db.conn.close()
            except Exception:
                pass
        return win_rate < win_rate_threshold
    except Exception:
        # When metrics cannot be read we err on the side of retraining to
        # recover from potential corruption.
        return True

MODEL_PATH = Path("retrieval_ranker.json")


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
        win_rate_threshold: float | None = None,
        roi_history_path: Path | str = "roi_history.db",
    ) -> None:
        # Ensure all services expose a context builder capable of refreshing
        # database weights.  This is now a required dependency so the scheduler
        # can refresh ranking weights without falling back to ``getattr`` at
        # runtime.
        self.services = list(services)
        for svc in self.services:
            try:
                cb = svc.context_builder  # type: ignore[attr-defined]
            except AttributeError as exc:  # pragma: no cover - validation
                raise AttributeError(
                    "All services must provide a `context_builder` with"
                    " `refresh_db_weights`."
                ) from exc
            if not hasattr(cb, "refresh_db_weights"):
                raise AttributeError(
                    "All services must provide a `context_builder` with"
                    " `refresh_db_weights`."
                )
        self.vector_db = Path(vector_db)
        self.metrics_db = Path(metrics_db)
        self.model_path = Path(model_path)
        self.interval = interval
        self.roi_tracker = roi_tracker
        self.roi_signal_threshold = roi_signal_threshold
        self.win_rate_threshold = win_rate_threshold
        self.roi_history_path = Path(roi_history_path)
        self.running = False
        self._thread: Optional[threading.Thread] = None
        # Track baseline ROI totals for ROITracker and VectorMetricsDB so that
        # cumulative deltas can trigger immediate retrains.
        self._tracker_roi_totals: dict[str, float] = {}
        self._vector_db_roi_totals: dict[str, float] = {}
        # Baseline of cumulative vector similarity per origin database.  This
        # allows the scheduler to react not only to ROI changes but also to
        # shifts in vector similarity scores which may indicate embedding
        # drift or search behaviour changes.
        self._vector_db_similarity_totals: dict[str, float] = {}
        # Accumulators for retrieval feedback events so short term ROI swings
        # or win-rate drops can trigger rapid retraining without waiting for
        # the periodic loop.
        self._roi_feedback_totals: dict[str, float] = {}
        self._feedback_events: int = 0
        self._feedback_wins: int = 0
        if self.roi_signal_threshold is None:
            env_thresh = os.getenv("RANKER_SCHEDULER_ROI_THRESHOLD")
            if env_thresh:
                try:
                    self.roi_signal_threshold = float(env_thresh)
                except Exception:
                    pass

        self.event_bus = event_bus
        if self.event_bus is None and UnifiedEventBus is not None:
            try:
                persist = os.getenv("RANKER_SCHEDULER_EVENT_LOG")
                rabbit = os.getenv("RANKER_SCHEDULER_RABBITMQ_HOST")
                self.event_bus = UnifiedEventBus(
                    persist_path=persist, rabbitmq_host=rabbit
                )
            except Exception:
                self.event_bus = None
        if self.event_bus is not None:
            try:
                self.event_bus.subscribe("patch_logger:outcome", self._handle_patch_outcome)
                self.event_bus.subscribe("retrieval:feedback", self._handle_retrieval_feedback)
            except Exception:
                pass

        if self.roi_tracker is not None:
            try:
                self.roi_tracker.load_history(str(self.roi_history_path))
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
    def _handle_retrieval_feedback(self, topic: str, event: object) -> None:
        """Handle retrieval feedback events and trigger retrain when needed."""

        if not isinstance(event, dict):
            return

        try:
            roi = float(event.get("roi", 0.0))
        except Exception:
            roi = 0.0
        win = bool(event.get("win"))
        origin = str(event.get("db") or event.get("origin") or "")

        self._feedback_events += 1
        if win:
            self._feedback_wins += 1
        if origin:
            total = self._roi_feedback_totals.get(origin, 0.0) + roi
            self._roi_feedback_totals[origin] = total
        else:
            total = roi

        triggered = False
        if self.roi_signal_threshold is not None and origin:
            if abs(self._roi_feedback_totals[origin]) >= self.roi_signal_threshold:
                triggered = True
                self._roi_feedback_totals[origin] = 0.0
        if (
            not triggered
            and self.win_rate_threshold is not None
            and self._feedback_events
        ):
            win_rate = self._feedback_wins / self._feedback_events
            if win_rate <= self.win_rate_threshold:
                triggered = True

        if not triggered:
            return

        threshold = (
            self.win_rate_threshold if self.win_rate_threshold is not None else 0.5
        )
        try:
            if needs_retrain(self.vector_db, threshold):
                self.retrain_and_reload()
        except Exception:
            logging.exception("ranking model retrain failed")
        finally:
            self._feedback_events = 0
            self._feedback_wins = 0
            self._roi_feedback_totals = {}

    # ------------------------------------------------------------------
    def _persist_model_path(self, new_path: Path, *, keep: int = 3) -> None:
        """Record *new_path* in the model registry and rotate old entries."""

        data = {"current": str(new_path), "history": []}
        if self.model_path.exists():
            try:
                with open(self.model_path, "r", encoding="utf-8") as fh:
                    loaded = json.load(fh)
                    if isinstance(loaded, dict):
                        data.update(loaded)
            except Exception:
                pass
        history = data.get("history", []) or []
        current = data.get("current")
        if current and current != str(new_path):
            history.insert(0, current)
        # Trim history and remove excess files
        for old in history[keep - 1 :]:
            try:
                Path(old).unlink()
            except Exception:
                pass
        history = history[: keep - 1]
        data["current"] = str(new_path)
        data["history"] = history
        try:
            with open(self.model_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def retrain_and_reload(self) -> None:
        """Retrain ranking model and notify services to reload."""
        if self.win_rate_threshold is not None:
            try:
                if not needs_retrain(self.vector_db, self.win_rate_threshold):
                    return
            except Exception:
                # If metrics cannot be read we continue with retraining to
                # recover from potential database issues.
                pass

        # Update reliability KPIs before training so latest values are joined
        compute_retriever_stats(self.metrics_db)

        # Recalculate ranking weights from accumulated metrics so training
        # reflects the latest ROI and safety signals.
        new_weights: dict[str, float] = {}
        try:
            db = VectorMetricsDB(self.vector_db)
            try:
                new_weights = db.recalc_ranking_weights()
            finally:
                db.conn.close()
        except Exception:
            logging.exception("ranking weight update failed")

        if self.roi_tracker is not None:
            try:
                base_roi = (
                    float(self.roi_tracker.roi_history[-1])
                    if self.roi_tracker.roi_history
                    else 0.0
                )
                _base, raroi, _ = self.roi_tracker.calculate_raroi(
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

        # Build the training dataframe directly from the metrics databases and
        # invoke ``retrieval_ranker`` to fit a new model.  This mirrors running
        # ``python retrieval_ranker.py train`` but avoids spawning a subprocess.
        try:
            df = build_dataset(vector_db=self.vector_db, patch_db=self.metrics_db)
        except Exception:
            logging.exception("failed to build training dataset")
            df = pd.DataFrame()
        trained = rr.train(df)
        tm = trained[0] if isinstance(trained, tuple) else trained
        model_dir = self.model_path.parent
        model_dir.mkdir(parents=True, exist_ok=True)
        model_file = model_dir / f"{self.model_path.stem}_{int(time.time())}.json"
        rr.save_model(tm, model_file)
        self._persist_model_path(model_file)

        # Load any persisted ranking weights so services can refresh their
        # ContextBuilder instances.  ``update_ranker`` stores these weights
        # under the ``weights`` key in ``retrieval_ranker.json``.
        weights: dict[str, float] = new_weights or {}
        if not weights:
            try:
                cfg = json.loads(self.model_path.read_text())
                if isinstance(cfg, dict):
                    w = cfg.get("weights")
                    if isinstance(w, dict):
                        weights = {str(k): float(v) for k, v in w.items()}
            except Exception:
                pass

        # Reload model, reliability scores and ranking weights in running services
        def _reload_all(svc: Any) -> None:
            try:
                if hasattr(svc, "reload_ranker_model"):
                    svc.reload_ranker_model(model_file)
                if hasattr(svc, "reload_reliability_scores"):
                    svc.reload_reliability_scores()
                if weights:
                    svc.context_builder.refresh_db_weights(weights)  # type: ignore[attr-defined]
            except Exception:
                return
            for dep in getattr(svc, "dependent_services", []) or []:
                _reload_all(dep)

        for svc in self.services:
            _reload_all(svc)

        if self.event_bus is not None:
            try:
                self.event_bus.publish(
                    "CognitionLayer.reload_ranker_model", {"path": str(model_file)}
                )
            except Exception:
                pass
            try:
                self.event_bus.publish("reload_reliability_scores", {})
            except Exception:
                pass
            # Notify listeners that the ranker has been updated so they can
            # reload models promptly.
            try:
                self.event_bus.publish(
                    "retrieval:ranker_updated", {"path": str(model_file)}
                )
            except Exception:
                pass

        # Snapshot ROI baselines after retraining so subsequent deltas are
        # measured relative to the freshly trained model.
        if self.roi_tracker is not None:
            try:
                self._tracker_roi_totals = {
                    origin: sum(float(d) for d in deltas)
                    for origin, deltas in self.roi_tracker.origin_db_delta_history.items()
                }
            except Exception:
                self._tracker_roi_totals = {}
        try:
            db = VectorMetricsDB(self.vector_db)
            try:
                cur = db.conn.execute(
                    """
                    SELECT db,
                           COALESCE(SUM(contribution),0) AS contrib,
                           COALESCE(SUM(similarity),0) AS sim
                      FROM vector_metrics
                     GROUP BY db
                    """
                )
                rows = cur.fetchall()
            finally:
                db.conn.close()
            self._vector_db_roi_totals = {
                str(origin): float(contrib or 0.0)
                for origin, contrib, _ in rows
            }
            self._vector_db_similarity_totals = {
                str(origin): float(sim or 0.0) for origin, _, sim in rows
            }
        except Exception:
            self._vector_db_roi_totals = {}
            self._vector_db_similarity_totals = {}

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
            if not self.running:
                time.sleep(base_interval)
                break
            while self.running:
                triggered = False
                if self.roi_signal_threshold is not None:
                    if self.roi_tracker is not None:
                        for origin, deltas in self.roi_tracker.origin_db_delta_history.items():
                            total = sum(float(d) for d in deltas)
                            last = self._tracker_roi_totals.get(origin, 0.0)
                            if abs(total - last) >= self.roi_signal_threshold:
                                triggered = True
                                break
                    if not triggered:
                        try:
                            db = VectorMetricsDB(self.vector_db)
                            try:
                                cur = db.conn.execute(
                                    """
                                    SELECT db,
                                           COALESCE(SUM(contribution),0) AS contrib,
                                           COALESCE(SUM(similarity),0) AS sim
                                      FROM vector_metrics
                                     GROUP BY db
                                    """
                                )
                                rows = cur.fetchall()
                            finally:
                                db.conn.close()
                            for origin, contrib, sim in rows:
                                origin = str(origin)
                                contrib = float(contrib or 0.0)
                                sim = float(sim or 0.0)
                                last_contrib = self._vector_db_roi_totals.get(origin, 0.0)
                                last_sim = self._vector_db_similarity_totals.get(origin, 0.0)
                                if (
                                    abs(contrib - last_contrib) >= self.roi_signal_threshold
                                    or abs(sim - last_sim) >= self.roi_signal_threshold
                                ):
                                    triggered = True
                                    break
                        except Exception:
                            pass
                    if triggered:
                        self.retrain_and_reload()
                        start = time.time()
                        continue
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


def schedule_ranking_model_retrain(
    services: Sequence[Any],
    *,
    interval: int = 86400,
    **kwargs: Any,
) -> RankingModelScheduler:
    """Start a background job that periodically retrains the ranking model."""

    sched = RankingModelScheduler(services, interval=interval, **kwargs)
    sched.start()
    return sched


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

