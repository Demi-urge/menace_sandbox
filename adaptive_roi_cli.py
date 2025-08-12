"""Train and use the adaptive ROI predictor."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence, Dict, Any

from .adaptive_roi_predictor import AdaptiveROIPredictor, load_training_data
from .adaptive_roi_dataset import build_dataset
from .roi_tracker import ROITracker


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
def _refresh(args: argparse.Namespace) -> None:
    """Periodically rebuild the dataset without retraining."""

    tracker = ROITracker()
    if args.history:
        tracker.load_history(args.history)

    while True:
        try:
            load_training_data(
                tracker,
                evolution_path=args.evolution_db,
                evaluation_path=args.evaluation_db,
                roi_events_path=args.roi_events_db,
                output_path=args.output_csv,
            )
            print(f"dataset refreshed -> {args.output_csv}")
        except Exception as exc:  # pragma: no cover
            print(f"refresh failed: {exc}")

        if args.once:
            break
        time.sleep(args.interval)


# ---------------------------------------------------------------------------
def _schedule(args: argparse.Namespace) -> None:
    """Periodically load training data and retrain the model."""

    tracker = ROITracker()
    if args.history:
        tracker.load_history(args.history)

    param_grid: Dict[str, Dict[str, Any]] | None = None
    if args.param_grid:
        param_grid = json.loads(args.param_grid)

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
                evaluation_path=args.evaluation_db,
                roi_events_path=args.roi_events_db,
                output_path=args.output_csv,
            )
            X, y, g, names = build_dataset(
                args.evolution_db,
                args.roi_db,
                args.evaluation_db,
                roi_events_path=args.roi_events_db,
                selected_features=selected,
                return_feature_names=True,
            )
            dataset = (X, y, g)
            predictor = AdaptiveROIPredictor(
                model_path=args.model,
                cv=args.cv,
                param_grid=param_grid,
                slope_threshold=args.slope_threshold,
                curvature_threshold=args.curvature_threshold,
            )
            predictor.train(
                dataset, cv=args.cv, param_grid=param_grid, feature_names=names
            )
            print(
                f"model retrained on {len(dataset[0])} samples -> {args.model}"
            )
        except Exception as exc:  # pragma: no cover
            print(f"retraining failed: {exc}")

        if args.once:
            break
        time.sleep(args.interval)


# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="sandbox_data/adaptive_roi.pkl",
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

    p_refresh = sub.add_parser(
        "refresh", help="periodically rebuild the dataset"
    )
    p_refresh.add_argument("--evolution-db", default="evolution_history.db")
    p_refresh.add_argument("--evaluation-db", default="evaluation_history.db")
    p_refresh.add_argument("--roi-events-db", default="roi_events.db")
    p_refresh.add_argument(
        "--history", default="sandbox_data/roi_history.json", help="Tracker history path"
    )
    p_refresh.add_argument(
        "--output-csv", default="sandbox_data/adaptive_roi.csv", help="CSV dump path"
    )
    p_refresh.add_argument(
        "--interval", type=int, default=3600, help="Seconds between refresh"
    )
    p_refresh.add_argument("--once", action="store_true", help="Run one cycle and exit")
    p_refresh.set_defaults(func=_refresh)

    p_sched = sub.add_parser(
        "schedule", help="periodically load data and retrain the model"
    )
    p_sched.add_argument("--evolution-db", default="evolution_history.db")
    p_sched.add_argument("--roi-db", default="roi.db")
    p_sched.add_argument("--evaluation-db", default="evaluation_history.db")
    p_sched.add_argument("--roi-events-db", default="roi_events.db")
    p_sched.add_argument(
        "--history", default="sandbox_data/roi_history.json", help="Tracker history path"
    )
    p_sched.add_argument(
        "--output-csv", default="sandbox_data/adaptive_roi.csv", help="CSV dump path"
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
    p_sched.set_defaults(func=_schedule)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

