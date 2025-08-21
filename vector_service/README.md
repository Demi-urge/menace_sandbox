# Vector Service

## Safety filtering

`ContextBuilder` and `Retriever` apply configurable safety thresholds:

- `max_alignment_severity`: skip vectors with `alignment_severity` above this value.
- `max_alerts`: skip vectors with more than this number of `semantic_alerts`.
- `license_denylist`: skip vectors whose `license` or fingerprint maps to a denylisted identifier.

Filtered vectors are counted by `PatchLogger` using the metric `patch_logger_vectors_total{risk="filtered"}`.

### Examples

With `max_alignment_severity=0.5`, `max_alerts=1` and `license_denylist={"GPL-3.0"}`:

- a vector with `alignment_severity=0.8` is dropped;
- a vector containing two `semantic_alerts` is dropped;
- a vector marked with license `GPL-3.0` is dropped.

## Retraining the ranker

`analytics/retrain_vector_ranker.py` rebuilds the ranking model from the latest
metrics, stores its path in `retrieval_ranker.json` and asks running
`CognitionLayer` instances to hot‑reload:

```bash
python -m analytics.retrain_vector_ranker --service myapp.layer:svc
```

Use `--interval 3600` to keep retraining hourly.

## Ranker scheduler

`analytics/ranker_scheduler.py` can run the retrainer on a timer. Set
environment variables and invoke the module to enable it:

```bash
export RANKER_SCHEDULER_INTERVAL=3600  # seconds between retrains (0 disables)
export RANKER_SCHEDULER_ROI_THRESHOLD=5  # optional ROI delta for immediate retrain
python -m analytics.ranker_scheduler
```

`RANKER_SCHEDULER_SERVICES` may list comma‑separated `module:attr` paths for
services exposing `reload_ranker_model`.
