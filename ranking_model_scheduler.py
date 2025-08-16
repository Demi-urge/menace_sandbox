from __future__ import annotations

"""Scheduler for periodic ranking model retraining.

This utility ties together the :mod:`retrieval_ranker` training utilities
with :func:`metrics_aggregator.compute_retriever_stats` so that the
ranking model and database win/regret rates stay up-to-date.  After
retraining the model it triggers a reload on any running services that
expose ``reload_ranker_model`` and ``reload_reliability_scores`` methods.
"""

from pathlib import Path
import threading
import time
from typing import Iterable, Sequence, Any, Optional

try:  # pragma: no cover - package-relative import
    from . import retrieval_ranker as rr
    from .metrics_aggregator import compute_retriever_stats
except Exception:  # pragma: no cover - fallback when executed directly
    import retrieval_ranker as rr  # type: ignore
    from metrics_aggregator import compute_retriever_stats  # type: ignore


class RankingModelScheduler:
    """Periodically retrain the ranking model and refresh reliability KPIs."""

    def __init__(
        self,
        services: Sequence[Any],
        *,
        vector_db: Path | str = "vector_metrics.db",
        metrics_db: Path | str = "metrics.db",
        model_path: Path | str = "retrieval_ranker.json",
        interval: int = 86400,
    ) -> None:
        self.services = list(services)
        self.vector_db = Path(vector_db)
        self.metrics_db = Path(metrics_db)
        self.model_path = Path(model_path)
        self.interval = interval
        self.running = False
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    def retrain_and_reload(self) -> None:
        """Retrain ranking model and notify services to reload."""
        # Update reliability KPIs before training so latest values are joined
        compute_retriever_stats(self.metrics_db)

        # Train model from latest vector metrics
        df = rr.load_training_data(
            vector_db=self.vector_db, patch_db=self.metrics_db
        )
        tm = rr.train(df)
        rr.save_model(tm, self.model_path)

        # Reload model and reliability scores in running services
        for svc in self.services:
            try:
                if hasattr(svc, "reload_ranker_model"):
                    svc.reload_ranker_model(self.model_path)
                if hasattr(svc, "reload_reliability_scores"):
                    svc.reload_reliability_scores()
            except Exception:
                continue

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        while self.running:
            self.retrain_and_reload()
            time.sleep(self.interval)

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

    p = argparse.ArgumentParser(description="Retrain ranking model periodically")
    p.add_argument("--vector-db", default="vector_metrics.db")
    p.add_argument("--metrics-db", default="metrics.db")
    p.add_argument("--model-path", default="retrieval_ranker.json")
    p.add_argument("--interval", type=int, default=86400)
    args = p.parse_args(list(argv) if argv is not None else None)

    sched = RankingModelScheduler([],
                                   vector_db=args.vector_db,
                                   metrics_db=args.metrics_db,
                                   model_path=args.model_path,
                                   interval=args.interval)
    sched.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        sched.stop()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

