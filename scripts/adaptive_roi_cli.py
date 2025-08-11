#!/usr/bin/env python3
"""Train and use the adaptive ROI predictor."""
from __future__ import annotations

import argparse
import json
from typing import Sequence

from adaptive_roi_predictor import AdaptiveROIPredictor
from adaptive_roi_dataset import build_dataset


# ---------------------------------------------------------------------------
def _train(args: argparse.Namespace) -> None:
    """Train a new predictor and persist it."""
    dataset = build_dataset(args.evolution_db, args.roi_db, args.evaluation_db)
    predictor = AdaptiveROIPredictor(model_path=args.model)
    predictor.train(dataset)
    print(f"model trained on {len(dataset[0])} samples -> {args.model}")


# ---------------------------------------------------------------------------
def _predict(args: argparse.Namespace) -> None:
    """Run prediction for ``features`` and print JSON result."""
    features = json.loads(args.features)
    predictor = AdaptiveROIPredictor(model_path=args.model)
    roi, growth = predictor.predict(features)
    print(json.dumps({"roi": roi, "growth": growth}))


# ---------------------------------------------------------------------------
def _retrain(args: argparse.Namespace) -> None:
    """Retrain an existing model with updated data."""
    dataset = build_dataset(args.evolution_db, args.roi_db, args.evaluation_db)
    predictor = AdaptiveROIPredictor(model_path=args.model)
    predictor.train(dataset)
    print(f"model retrained on {len(dataset[0])} samples -> {args.model}")


# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="adaptive_roi_model.pkl", help="Model path")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="train a new model")
    p_train.add_argument("--evolution-db", default="evolution_history.db")
    p_train.add_argument("--roi-db", default="roi.db")
    p_train.add_argument("--evaluation-db", default="evaluation_history.db")
    p_train.set_defaults(func=_train)

    p_predict = sub.add_parser("predict", help="predict ROI for a feature sequence")
    p_predict.add_argument("features", help="JSON encoded feature matrix")
    p_predict.set_defaults(func=_predict)

    p_retrain = sub.add_parser("retrain", help="retrain the model with latest data")
    p_retrain.add_argument("--evolution-db", default="evolution_history.db")
    p_retrain.add_argument("--roi-db", default="roi.db")
    p_retrain.add_argument("--evaluation-db", default="evaluation_history.db")
    p_retrain.set_defaults(func=_retrain)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
