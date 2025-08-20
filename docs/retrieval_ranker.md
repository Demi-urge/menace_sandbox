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
