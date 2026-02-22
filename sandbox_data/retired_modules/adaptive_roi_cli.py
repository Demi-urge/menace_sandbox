"""Train and use the adaptive ROI predictor.

The underlying :class:`~menace_sandbox.adaptive_roi_predictor.AdaptiveROIPredictor`
supports online learning.  New samples can be appended without retraining
from scratch by calling ``predictor.update(X, y, g)`` which performs a
``partial_fit`` when supported by the model.
"""
from __future__ import annotations

import argparse
import json
import time
import logging
from pathlib import Path
from typing import Sequence, Dict, Any
import numpy as np

from .adaptive_roi_predictor import AdaptiveROIPredictor, load_training_data
from .adaptive_roi_dataset import build_dataset
from .roi_tracker import ROITracker
from .truth_adapter import TruthAdapter
from .composite_workflow_scorer import CompositeWorkflowScorer
from .roi_results_db import ROIResultsDB
from context_builder_util import create_context_builder
import db_router
from .dynamic_path_router import resolve_path
from .self_coding_thresholds import update_thresholds as save_sc_thresholds
from .data_bot import DataBot
from .unified_event_bus import UnifiedEventBus


# ---------------------------------------------------------------------------
def _train(args: argparse.Namespace) -> None:
    """Train a new predictor and persist it."""
    selected = None
    if args.selected_features:
        meta_path = Path(args.model).with_suffix(".meta.json")
        try:
            data = json.loads(meta_path.read_text())
            sel = data.get("selected_features")
            if isinstance(sel, list) and sel:
                selected = [str(s) for s in sel]
        except Exception:
            selected = None
    router = db_router.DBRouter(
        "adaptive_cli", str(args.evaluation_db), str(args.evaluation_db)
    )
    X, y, g, names = build_dataset(
        args.evolution_db,
        router=router,
        selected_features=selected,
        return_feature_names=True,
    )
    dataset = (X, y, g)
    param_grid: Dict[str, Dict[str, Any]] | None = None
    if args.param_grid:
        param_grid = json.loads(args.param_grid)
    predictor = AdaptiveROIPredictor(
        model_path=args.model,
        cv=args.cv,
        param_grid=param_grid,
        slope_threshold=args.slope_threshold,
        curvature_threshold=args.curvature_threshold,
    )
    predictor.train(dataset, cv=args.cv, param_grid=param_grid, feature_names=names)
    try:
        TruthAdapter().fit(X, y)
    except Exception:
        logging.getLogger(__name__).exception("truth adapter training failed")
    print(f"model trained on {len(dataset[0])} samples -> {args.model}")
    if predictor.validation_scores:
        print("validation MAE:")
        for name, score in predictor.validation_scores.items():
            print(f"  {name}: {score:.4f}")
    if predictor.best_params and predictor.best_score is not None:
        params = {k: v for k, v in predictor.best_params.items() if k != "model"}
        print(
            f"best model: {predictor.best_params['model']} "
            f"(MAE={predictor.best_score:.4f})"
        )
        if params:
            print(f"best params: {params}")


# ---------------------------------------------------------------------------
def _predict(args: argparse.Namespace) -> None:
    """Run prediction for ``features`` and print JSON result."""
    features = json.loads(args.features)
    predictor = AdaptiveROIPredictor(
        model_path=args.model,
        slope_threshold=args.slope_threshold,
        curvature_threshold=args.curvature_threshold,
    )
    roi_seq, growth, conf = predictor.predict(features, horizon=args.horizon)
    try:
        arr = np.asarray(features, dtype=float)
        realish, low_conf = TruthAdapter().predict(arr)
        roi_seq = realish.tolist()
        if low_conf:
            print("truth adapter low confidence; retrain suggested")
    except Exception:
        logging.getLogger(__name__).exception("truth adapter predict failed")
    print(json.dumps({"roi": roi_seq, "growth": growth, "confidence": conf}))


# ---------------------------------------------------------------------------
def _retrain(args: argparse.Namespace) -> None:
    """Retrain an existing model with updated data."""
    selected = None
    if args.selected_features:
        meta_path = Path(args.model).with_suffix(".meta.json")
        try:
            data = json.loads(meta_path.read_text())
            sel = data.get("selected_features")
            if isinstance(sel, list) and sel:
                selected = [str(s) for s in sel]
        except Exception:
            selected = None
    X, y, g, names = build_dataset(
        args.evolution_db,
        args.roi_db,
        args.evaluation_db,
        selected_features=selected,
        return_feature_names=True,
    )
    dataset = (X, y, g)
    param_grid: Dict[str, Dict[str, Any]] | None = None
    if args.param_grid:
        param_grid = json.loads(args.param_grid)
    predictor = AdaptiveROIPredictor(
        model_path=args.model,
        cv=args.cv,
        param_grid=param_grid,
        slope_threshold=args.slope_threshold,
        curvature_threshold=args.curvature_threshold,
    )
    predictor.train(dataset, cv=args.cv, param_grid=param_grid, feature_names=names)
    try:
        TruthAdapter().fit(X, y)
    except Exception:
        logging.getLogger(__name__).exception("truth adapter training failed")
    print(f"model retrained on {len(dataset[0])} samples -> {args.model}")
    if predictor.validation_scores:
        print("validation MAE:")
        for name, score in predictor.validation_scores.items():
            print(f"  {name}: {score:.4f}")
    if predictor.best_params and predictor.best_score is not None:
        params = {k: v for k, v in predictor.best_params.items() if k != "model"}
        print(
            f"best model: {predictor.best_params['model']} "
            f"(MAE={predictor.best_score:.4f})"
        )
        if params:
            print(f"best params: {params}")


# ---------------------------------------------------------------------------
def _thresholds(args: argparse.Namespace) -> None:
    """Adjust self-coding thresholds for a bot and broadcast update."""
    save_sc_thresholds(
        args.bot,
        roi_drop=args.roi_drop,
        error_increase=args.error_increase,
        test_failure_increase=args.test_failure_increase,
    )
    bus = UnifiedEventBus()
    try:
        DataBot(event_bus=bus, start_server=False).reload_thresholds(args.bot)
    finally:
        bus.close()
    print(f"thresholds updated for {args.bot}")


# ---------------------------------------------------------------------------
def _refresh(args: argparse.Namespace) -> None:
    """Periodically rebuild the dataset without retraining."""

    tracker = ROITracker()
    if args.history:
        tracker.load_history(args.history)
    router = db_router.DBRouter(
        "adaptive_cli", str(args.evaluation_db), str(args.evaluation_db)
    )

    while True:
        try:
            load_training_data(
                tracker,
                evolution_path=args.evolution_db,
                roi_events_path=args.roi_events_db,
                output_path=args.output_csv,
                router=router,
            )
            print(f"dataset refreshed -> {args.output_csv}")
        except Exception as exc:  # pragma: no cover
            print(f"refresh failed: {exc}")

        if args.once:
            break
        time.sleep(args.interval)


# ---------------------------------------------------------------------------
def _scorecard(args: argparse.Namespace) -> None:
    """Generate a scenario scorecard for ``workflow_id``."""

    tracker = ROITracker()
    if args.history:
        tracker.load_history(args.history)

    scenarios = args.scenarios.split(",") if args.scenarios else None
    card = tracker.generate_scenario_scorecard(args.workflow_id, scenarios)
    text = json.dumps(card, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text)
        print(f"scorecard saved -> {args.output}")
    else:
        print(text)


# ---------------------------------------------------------------------------
def _workflow_eval(args: argparse.Namespace) -> None:
    """Evaluate a workflow and persist metrics, printing the run id."""

    presets = json.loads(args.presets) if args.presets else None
    scorer = CompositeWorkflowScorer(
        tracker=ROITracker(), results_db=ROIResultsDB(args.db_path)
    )
    builder = create_context_builder()
    try:
        builder.refresh_db_weights()
    except Exception:  # pragma: no cover - best effort refresh
        pass
    scorer.evaluate(args.workflow_id, env_presets=presets, context_builder=builder)
    cur = scorer.results_db.conn.cursor()
    row = cur.execute(
        "SELECT run_id FROM workflow_results WHERE workflow_id=? ORDER BY timestamp DESC LIMIT 1",
        (args.workflow_id,),
    ).fetchone()
    if row:
        print(row[0])


# ---------------------------------------------------------------------------
def _schedule(args: argparse.Namespace) -> None:
    """Periodically load training data and retrain the model."""
    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("adaptive_roi_schedule")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(log_path)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
        )
        logger.addHandler(handler)

    tracker = ROITracker()
    if args.history:
        tracker.load_history(args.history)

    router = db_router.DBRouter(
        "adaptive_cli", str(args.evaluation_db), str(args.evaluation_db)
    )

    param_grid: Dict[str, Dict[str, Any]] | None = None
    if args.param_grid:
        param_grid = json.loads(args.param_grid)
    adapter = TruthAdapter()

    while True:
        try:
            selected = None
            if args.selected_features:
                meta_path = Path(args.model).with_suffix(".meta.json")
                try:
                    data = json.loads(meta_path.read_text())
                    sel = data.get("selected_features")
                    if isinstance(sel, list) and sel:
                        selected = [str(s) for s in sel]
                except Exception:
                    selected = None
            load_training_data(
                tracker,
                evolution_path=args.evolution_db,
                roi_events_path=args.roi_events_db,
                output_path=args.output_csv,
                router=router,
            )
            logger.info("training data loaded")
            X, y, g, names = build_dataset(
                args.evolution_db,
                router=router,
                roi_events_path=args.roi_events_db,
                selected_features=selected,
                return_feature_names=True,
            )
            dataset = (X, y, g)
            try:
                adapter.fit(X, y)
            except Exception:
                logger.exception("truth adapter training failed")
            slope_thr = getattr(args, "slope_threshold", None)
            curv_thr = getattr(args, "curvature_threshold", None)
            from .adaptive_roi_predictor import AdaptiveROIPredictor as Predictor
            try:
                predictor = Predictor(
                    model_path=args.model,
                    cv=args.cv,
                    param_grid=param_grid,
                    slope_threshold=slope_thr,
                    curvature_threshold=curv_thr,
                )
            except TypeError:  # pragma: no cover - fallback for simplified stubs
                predictor = Predictor()
            try:
                predictor.train(
                    dataset, cv=args.cv, param_grid=param_grid, feature_names=names
                )
            except TypeError:  # pragma: no cover - fallback for simplified stubs
                predictor.train()
            msg = f"model retrained on {len(dataset[0])} samples -> {args.model}"
            print(msg)
            logger.info(msg)
            val_scores = getattr(predictor, "validation_scores", {}) or {}
            if val_scores:
                for name, score in val_scores.items():
                    logger.info("validation MAE %s: %.4f", name, score)
            best_params = getattr(predictor, "best_params", None)
            best_score = getattr(predictor, "best_score", None)
            if best_params and best_score is not None:
                params = {k: v for k, v in best_params.items() if k != "model"}
                logger.info(
                    "best model %s (MAE=%.4f)",
                    best_params.get("model"),
                    best_score,
                )
                if params:
                    logger.info("best params: %s", params)
        except Exception as exc:  # pragma: no cover
            logger.exception("retraining failed: %s", exc)
            print(f"retraining failed: {exc}")

        if args.once:
            break
        time.sleep(args.interval)


# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=str(Path(resolve_path("sandbox_data")) / "adaptive_roi.pkl"),
        help="Model path",
    )
    parser.add_argument("--slope-threshold", type=float, default=None, help="Slope threshold")
    parser.add_argument(
        "--curvature-threshold", type=float, default=None, help="Curvature threshold"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="train a new model")
    p_train.add_argument("--evolution-db", default="evolution_history.db")
    p_train.add_argument("--roi-db", default="roi.db")
    p_train.add_argument("--evaluation-db", default="evaluation_history.db")
    p_train.add_argument("--cv", type=int, default=3, help="Cross-validation folds")
    p_train.add_argument(
        "--param-grid",
        default=None,
        help="JSON encoded parameter grid for hyperparameter tuning",
    )
    p_train.add_argument(
        "--selected-features",
        action="store_true",
        help="Restrict training to features listed in the model's meta file",
    )
    p_train.set_defaults(func=_train)

    p_predict = sub.add_parser("predict", help="predict ROI for a feature sequence")
    p_predict.add_argument("features", help="JSON encoded feature matrix")
    p_predict.add_argument(
        "--horizon", type=int, default=None, help="Number of steps to forecast"
    )
    p_predict.set_defaults(func=_predict)

    p_retrain = sub.add_parser("retrain", help="retrain the model with latest data")
    p_retrain.add_argument("--evolution-db", default="evolution_history.db")
    p_retrain.add_argument("--roi-db", default="roi.db")
    p_retrain.add_argument("--evaluation-db", default="evaluation_history.db")
    p_retrain.add_argument("--cv", type=int, default=3, help="Cross-validation folds")
    p_retrain.add_argument(
        "--param-grid",
        default=None,
        help="JSON encoded parameter grid for hyperparameter tuning",
    )
    p_retrain.add_argument(
        "--selected-features",
        action="store_true",
        help="Restrict training to features listed in the model's meta file",
    )
    p_retrain.set_defaults(func=_retrain)

    p_thresh = sub.add_parser(
        "threshold", help="update self-coding thresholds for a bot"
    )
    p_thresh.add_argument("bot", help="Bot name")
    p_thresh.add_argument(
        "--roi-drop", type=float, default=None, help="ROI drop trigger"
    )
    p_thresh.add_argument(
        "--error-increase",
        type=float,
        default=None,
        help="Allowed error rate increase",
    )
    p_thresh.add_argument(
        "--test-failure-increase",
        type=float,
        default=None,
        help="Allowed increase in test failures",
    )
    p_thresh.set_defaults(func=_thresholds)

    p_refresh = sub.add_parser(
        "refresh", help="periodically rebuild the dataset"
    )
    p_refresh.add_argument("--evolution-db", default="evolution_history.db")
    p_refresh.add_argument("--evaluation-db", default="evaluation_history.db")
    p_refresh.add_argument("--roi-events-db", default="roi_events.db")
    p_refresh.add_argument(
        "--history",
        default=str(Path(resolve_path("sandbox_data")) / "roi_history.json"),
        help="Tracker history path",
    )
    p_refresh.add_argument(
        "--output-csv",
        default=str(Path(resolve_path("sandbox_data")) / "adaptive_roi.csv"),
        help="CSV dump path",
    )
    p_refresh.add_argument(
        "--interval", type=int, default=3600, help="Seconds between refresh"
    )
    p_refresh.add_argument("--once", action="store_true", help="Run one cycle and exit")
    p_refresh.set_defaults(func=_refresh)

    p_card = sub.add_parser("scorecard", help="generate scenario scorecard")
    p_card.add_argument("workflow_id", help="Workflow identifier")
    p_card.add_argument(
        "--scenarios",
        default=None,
        help="Comma separated scenario names; defaults to standard presets",
    )
    p_card.add_argument(
        "--history",
        default=str(Path(resolve_path("sandbox_data")) / "roi_history.json"),
        help="Tracker history path",
    )
    p_card.add_argument("--output", default=None, help="Write scorecard JSON to file")
    p_card.set_defaults(func=_scorecard)

    p_eval = sub.add_parser("workflow-eval", help="evaluate a workflow and store metrics")
    p_eval.add_argument("workflow_id", help="Workflow identifier")
    p_eval.add_argument(
        "--presets",
        default=None,
        help="JSON encoded environment presets",
    )
    p_eval.add_argument(
        "--db-path", default="roi_results.db", help="ROI results database path"
    )
    p_eval.set_defaults(func=_workflow_eval)

    p_sched = sub.add_parser(
        "schedule", help="periodically load data and retrain the model"
    )
    p_sched.add_argument("--evolution-db", default="evolution_history.db")
    p_sched.add_argument("--roi-db", default="roi.db")
    p_sched.add_argument("--evaluation-db", default="evaluation_history.db")
    p_sched.add_argument("--roi-events-db", default="roi_events.db")
    p_sched.add_argument(
        "--history",
        default=str(Path(resolve_path("sandbox_data")) / "roi_history.json"),
        help="Tracker history path",
    )
    p_sched.add_argument(
        "--output-csv",
        default=str(Path(resolve_path("sandbox_data")) / "adaptive_roi.csv"),
        help="CSV dump path",
    )
    p_sched.add_argument(
        "--interval", type=int, default=3600, help="Seconds between retraining"
    )
    p_sched.add_argument("--once", action="store_true", help="Run one cycle and exit")
    p_sched.add_argument("--cv", type=int, default=3, help="Cross-validation folds")
    p_sched.add_argument(
        "--param-grid",
        default=None,
        help="JSON encoded parameter grid for hyperparameter tuning",
    )
    p_sched.add_argument(
        "--selected-features",
        action="store_true",
        help="Restrict training to features listed in the model's meta file",
    )
    p_sched.add_argument(
        "--log-path",
        default=str(Path(resolve_path("sandbox_data")) / "adaptive_roi_schedule.log"),
        help="Log file path",
    )
    p_sched.set_defaults(func=_schedule)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
