from __future__ import annotations

"""CLI for retraining and evaluating the adaptive ROI predictor.

The command loads the latest evolution and evaluation logs, retrains the
:class:`AdaptiveROIPredictor` model and reports a simple hold-out evaluation
score.  It can be executed periodically (e.g. via cron) to keep the model in
sync with newly collected data.
"""

from pathlib import Path
import argparse
import pickle
from typing import Sequence

import numpy as np

from .adaptive_roi_dataset import load_adaptive_roi_dataset
from .adaptive_roi_predictor import GradientBoostingRegressor, LinearRegression


# ---------------------------------------------------------------------------
def train_model(evolution_db: str | Path, evaluation_db: str | Path):
    """Return a freshly trained model and validation MSE.

    The dataset is split 80/20 into training and validation segments.  ``None``
    is returned when insufficient data is available or no regression backend can
    be imported.
    """

    X, y, _ = load_adaptive_roi_dataset(evolution_db, evaluation_db)
    if not X.size or not y.size:
        return None

    if GradientBoostingRegressor is not None:
        model = GradientBoostingRegressor(random_state=0)
    elif LinearRegression is not None:
        model = LinearRegression()
    else:  # pragma: no cover - extremely minimal environment
        return None

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    split = max(1, int(len(X) * 0.8))
    train_idx, test_idx = idx[:split], idx[split:]
    model.fit(X[train_idx], y[train_idx])
    mse = float(np.mean((model.predict(X[test_idx]) - y[test_idx]) ** 2)) if len(test_idx) else 0.0
    return model, mse


# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evolution-db", default="evolution_history.db", help="Path to evolution history database")
    parser.add_argument("--evaluation-db", default="evaluation_history.db", help="Path to evaluation history database")
    parser.add_argument("--model-out", default="adaptive_roi_model.pkl", help="Where to store the trained model")
    args = parser.parse_args(argv)

    result = train_model(args.evolution_db, args.evaluation_db)
    if result is None:
        print("No data available or no suitable model backend installed.")
        return 1

    model, mse = result
    with open(args.model_out, "wb") as fh:
        pickle.dump(model, fh)
    print(f"Model saved to {args.model_out}; validation MSE={mse:.4f}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
