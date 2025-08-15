# Retrieval Ranker

`analytics/retrieval_ranker.py` trains a lightweight model that estimates the
chance a retrieved record will lead to a successful patch.  Training requires
two JSONL datasets:

* `analytics/retrieval_outcomes.jsonl` – retrieval statistics including
  ``origin_db``, ``age``, ``similarity``, ``roi_delta``, execution frequency,
  prior hit counts and win/regret rates.
* `analytics/patch_outcomes.jsonl` – patch outcome labels keyed by
  ``session_id`` with a boolean ``win`` column.

Run a single training pass with:

```bash
python analytics/retrieval_ranker.py \
    --stats-path analytics/retrieval_outcomes.jsonl \
    --labels-path analytics/patch_outcomes.jsonl \
    --model-path analytics/retrieval_ranker.pkl
```

Periodic retraining can be scheduled by supplying ``--interval`` in seconds:

```bash
python analytics/retrieval_ranker.py --interval 3600
```

The generated ``retrieval_ranker.pkl`` file contains the fitted model and the
feature names used during training.

