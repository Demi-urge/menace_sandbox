# Retrieval Ranker

`retrieval_ranker.py` trains a lightweight model that scores retrieval
results.  The training data is assembled by
`retrieval_training_dataset.build_dataset` from the metric databases.

## Training

Run a single training pass with the default database locations:

```bash
python retrieval_ranker.py train \
    --vector-db vector_metrics.db \
    --patch-db metrics.db \
    --model-path analytics/retrieval_ranker.model
```

The produced `analytics/retrieval_ranker.model` file contains the model
coefficients and the feature names used during training.  It is compatible with
`menace.universal_retriever.load_ranker`.

## Nightly retraining

`analytics/retrain_vector_ranker.py` retrains the model using the latest
`vector_metrics.db` and patch history in `metrics.db`, stores the updated
weights under `analytics/retrieval_ranker.model` and invokes
`UniversalRetriever.reload_ranker_model` on running services so they pick up the
new model.

This script runs nightly via a scheduled workflow but can also be executed
manually:

```bash
python analytics/retrain_vector_ranker.py \
    --vector-db path/to/vector_metrics.db \
    --patch-db path/to/metrics.db \
    --service module:service_instance
```

## Scheduled retrain with hot reload

`RankingModelScheduler` exposes a CLI for periodic retraining and hot reloading
of running services.  Pass one or more service import paths via `--service` and
the scheduler will invoke `reload_ranker_model` on each after a successful
retrain:

```bash
python -m menace_sandbox.ranking_model_scheduler \
    --vector-db vector_metrics.db \
    --metrics-db metrics.db \
    --model-path retrieval_ranker.json \
    --service mymodule:layer \
    --interval 3600
```

In this example `mymodule:layer` refers to a variable holding a
`CognitionLayer` instance.  The scheduler retrains the model every hour and
triggers a hot reload on that layer and any dependent services.

Environment variables can control its behaviour:

* `RANKER_SCHEDULER_ROI_THRESHOLD` – cumulative per-origin ROI delta that
  triggers an immediate retrain when crossed.
* `RANKER_SCHEDULER_RISK_THRESHOLD` – cumulative per-origin risk delta that
  triggers an immediate retrain when crossed.
* `RANKER_SCHEDULER_EVENT_LOG` – optional SQLite file for persisting
  `UnifiedEventBus` events.
* `RANKER_SCHEDULER_RABBITMQ_HOST` – RabbitMQ host for mirroring bus events.

When thresholds are provided the scheduler listens on `roi:update` and
accumulates ROI and risk deltas per origin.
