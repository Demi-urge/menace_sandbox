# Learning Engine

`LearningEngine` trains a simple model from historical workflow data and can run
cross-validation to measure predictive accuracy.  It pulls pathway statistics
from `PathwayDB` and uses `MenaceMemoryManager` to embed stored action traces.
The dataset combines frequency, execution time, ROI and myelination score with
an average embedding value.  Predictions can be used to guess whether a new
pathway will succeed.

```python
from menace.neuroplasticity import PathwayDB
from menace.menace_memory_manager import MenaceMemoryManager
from menace.learning_engine import LearningEngine

pdb = PathwayDB("pathways.db")
mm = MenaceMemoryManager("mem.db")
engine = LearningEngine(pdb, mm)
engine.train()
prob = engine.predict_success(1.0, 1.0, 1.0, 1.0, "actions")
```

### Transformer Models

Pass `model="transformer"` or `model="bert"` to use a lightweight
`transformers` model for text classification.  In this mode only the action
text is used and scikit‑learn is not required.

```python
engine = LearningEngine(pdb, mm, model="transformer")
engine.train()
score = engine.evaluate()
```

The engine is optional but can be supplied to `SelfImprovementEngine` so
that every cycle retrains on the latest data.

## Incremental Updates

When `PathwayDB` is initialised with a `UnifiedEventBus`, every call to
`log()` publishes a `"pathway:new"` event.  `SelfImprovementEngine` used to
listen directly for these events.  Now `SelfLearningCoordinator` subscribes to
`"memory:new"`, `"code:new"`, `"workflows:new"` and `"pathway:new"` topics and
performs automatic incremental training by calling `partial_train()` on the
relevant learning engines.  This keeps the models up to date with new memory
entries, new code snippets, workflow plans and freshly logged pathways without
running full training cycles.

## Self-Learning Service

To run the incremental training loop as a standalone process simply execute:

```bash
python -m menace.self_learning_service
```

This script initialises a `UnifiedEventBus`, all databases and learning engines
then starts `SelfLearningCoordinator` which listens for `memory:new`, `code:new`,
`workflows:new` and `pathway:new` events. It runs indefinitely until
interrupted.

## Evaluating Models

Call `evaluate()` on a learning engine to run cross-validation and a small
hold‑out test split.  Results are stored in `evaluation_history` for later
analysis.

```python
score = engine.evaluate()
print(score["cv_score"], score["holdout_score"])
```

### Automated Model Selection

Use `auto_train()` to try multiple models and keep the best performer.  This
also normalizes numeric features so different scales do not skew results.

```python
best = engine.auto_train(["logreg", "nn"])
print("best model", best)
```

### Metrics Export

When `persist_path` is provided the results of each `evaluate()` call are saved
to disk.  They can be pushed to a Prometheus server by combining
``metrics_exporter.start_metrics_server`` with periodic evaluations.

```python
from menace.metrics_exporter import start_metrics_server

start_metrics_server(8001)  # expose metrics on http://localhost:8001
```

Add the ``learning_cv_score`` and ``learning_holdout_score`` gauges to a
Grafana dashboard by configuring Prometheus as a data source and creating two
new panels pointing to these metric names.
