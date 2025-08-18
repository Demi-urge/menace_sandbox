# TruthAdapter

The TruthAdapter calibrates ROI predictions and tracks feature drift so the sandbox can detect when the model's output is unreliable.

## Purpose

It learns a regression model to align predicted ROI with observed results and records feature statistics for drift checks. When distributions shift beyond a threshold, the adapter raises a low-confidence flag.

## Training and Low-confidence Flags

Train the adapter on live ROI metrics or shadow evaluation data:

```python
from truth_adapter import TruthAdapter
adapter = TruthAdapter()
adapter.fit(X_live, y_live)  # or adapter.fit(X_shadow, y_shadow)
```

During operation, update drift statistics and retrieve calibrated values. `predict` returns both predictions and a flag indicating if drift has pushed the model into a low-confidence state:

```python
drift = adapter.check_drift(X_recent)
preds, low_conf = adapter.predict(X_recent)
if low_conf:
    # schedule retraining or inspect data
    ...
```

A true low-confidence flag signals the adapter no longer trusts its calibration.

## Persistence

Model parameters and metadata persist under `sandbox_data/truth_adapter.pkl` with a companion JSON file `sandbox_data/truth_adapter.pkl.meta.json` containing feature statistics and timestamps:

```
sandbox_data/
├── truth_adapter.pkl
└── truth_adapter.pkl.meta.json
```

Persisted metadata helps audit drift history across sandbox runs.
