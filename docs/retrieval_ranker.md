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

