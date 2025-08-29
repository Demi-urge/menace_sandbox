# TruthAdapter

The TruthAdapter calibrates ROI predictions and tracks feature drift so the sandbox can detect when the model's output is unreliable.

## Purpose

It learns a regression model to align predicted ROI with observed results and records feature statistics for drift checks. When distributions shift beyond a threshold, the adapter raises a low-confidence flag.

## Configuration

TruthAdapter is enabled when `SandboxSettings.enable_truth_calibration` or the
`ENABLE_TRUTH_CALIBRATION` environment variable is `True` and can be disabled by
setting `ENABLE_TRUTH_CALIBRATION=0`. By default the adapter saves its model and
drift metadata to `sandbox_data/truth_adapter.pkl`; pass
`TruthAdapter(model_path=...)` to use a different location. Choose the
underlying regression model via the `model_type` argument, for example
`TruthAdapter(model_type="xgboost")` if XGBoost is installed, otherwise a ridge
regressor is used.

Drift thresholds default to ``0.25`` for PSI and ``0.2`` for the KS statistic.
They can be overridden explicitly:

```python
TruthAdapter(psi_threshold=0.3, ks_threshold=0.25)
```

or configured globally via ``SandboxSettings`` with the ``psi_threshold`` and
``ks_threshold`` fields.

Hyperparameters for the underlying models can be provided through
`ridge_params` and `xgb_params`:

```python
TruthAdapter(model_type="ridge", ridge_params={"alpha": 2.0})
TruthAdapter(model_type="xgboost", xgb_params={"learning_rate": 0.1, "max_depth": 4})
```

## Workflow and Input Data

The adapter consumes **sandbox metrics** as feature vectors and a **profit proxy** such as ROI or revenue as the target. A typical workflow:

1. Gather recent sandbox metrics `X` and corresponding profit proxy values `y`.
2. Fit the adapter on this data to calibrate predictions and capture baseline feature statistics.
3. During inference, call `predict` on new metrics to obtain calibrated outputs while `check_drift` evaluates distribution shifts and returns drift metrics.

## Training and Low-confidence Flags
Train the adapter on live ROI metrics or shadow evaluation data. `fit` expects `X` to be a 2‑D array of sandbox metrics and `y` to be a 1‑D array containing the profit proxy:

```python
from truth_adapter import TruthAdapter
adapter = TruthAdapter()
adapter.fit(X_live, y_live)  # or adapter.fit(X_shadow, y_shadow)
```

Enable simple cross-validation to automatically select between ridge and
XGBoost (when both are available):

```python
adapter.fit(X_live, y_live, cross_validate=True)
```

During operation, update drift statistics and retrieve calibrated values. `predict` returns both predictions and a flag indicating if drift has pushed the model into a low-confidence state:

```python
metrics, drift = adapter.check_drift(X_recent)
preds, low_conf = adapter.predict(X_recent)
if low_conf:
    # schedule retraining or inspect data
    ...
```

A true low-confidence flag signals the adapter no longer trusts its calibration.

## Drift Warnings and Retraining

`check_drift` computes Population Stability Index and Kolmogorov–Smirnov statistics for each feature and returns them alongside a boolean drift indicator. When either metric exceeds its threshold, `metadata["drift_flag"]` and `metadata["needs_retrain"]` become `True`, and `predict` returns `low_conf=True`. Gather fresh `X` and `y` samples and call `fit` again to retrain and clear the warning.

For automated retraining the `SelfImprovementEngine` exposes a CLI:

```bash
python self_improvement.py fit-truth-adapter live.npz shadow.npz
```

This command merges live and shadow datasets and overwrites the persisted model
at `sandbox_data/truth_adapter.pkl`.

## Persistence

Model parameters and metadata persist under `sandbox_data/truth_adapter.pkl`. The metadata records feature statistics, drift metrics and the timestamp of the last retraining, enabling audits of drift history across sandbox runs.

## Calibration Workflow

Sandbox components that consume sandbox ROI now invoke the adapter to replace raw
values with calibrated predictions.  The `self_improvement` and
`action_planner` both run ROI figures through `TruthAdapter.predict` before
recording them via `ROITracker`.  The tracker persists the adapter's metadata so
dashboard endpoints such as `/roi_data` can surface `needs_retrain` flags and
`last_retrained` timestamps.

## Optional Dependencies

The adapter works with scikit-learn's `Ridge` by default. Installing the following packages enables additional features:

- **XGBoost** for gradient boosting models. Install with `pip install xgboost` and instantiate via `TruthAdapter(model_type="xgboost")`.
- **SciPy** for the Kolmogorov–Smirnov drift test. Install with `pip install scipy`. Without SciPy a histogram-based fallback is used.
- **joblib** for faster serialization. Install with `pip install joblib`.
