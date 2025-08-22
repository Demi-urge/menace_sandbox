# Vector Service

## Cognition Flow

`vectorizer.py` turns raw records into embeddings. `retriever.py` searches
existing vectors, and `context_builder.py` assembles a JSON context from the
top hits. `cognition_layer.py` orchestrates this flow:

1. `vectorizer.py` encodes new records and stores their vectors.
2. `retriever.py` ranks the stored vectors for a query.
3. `context_builder.py` builds a JSON payload and notes each origin database.
4. `cognition_layer.py` keeps session state and hands it off to `patch_logger.py`.
5. `patch_logger.py` records whether the patch worked, updating metrics.
6. `roi_tracker.py` (optional) aggregates ROI deltas per origin and pushes them
   back into the ranker so future retrievals favour high‑ROI sources.

### Example

```python
from roi_tracker import ROITracker  # optional
from vector_service.cognition_layer import CognitionLayer

tracker = ROITracker()
layer = CognitionLayer(roi_tracker=tracker)

ctx, session_id = layer.query("How can I fix latency?")
# ...apply patch based on ctx...
layer.record_patch_outcome(session_id, True, contribution=1.0)
# ranking weights now reflect ROI feedback
```

The final call updates ROI metrics and adjusts ranking weights.

## Feedback cycle

The service closes the loop on every patch by feeding outcomes back into the
retrieval ranker:

1. **Retrieval** – `retriever.py` selects candidate vectors.
2. **Context build** – `context_builder.py` packs the selected vectors into a
   JSON payload.
3. **Patch logging** – [`cognition_layer.py`](cognition_layer.py) forwards the
   contributors to [`patch_logger.py`](patch_logger.py), which uses
   [`patch_safety.py`](patch_safety.py) to score risk.
4. **ROI/risk updates** – `PatchLogger` records success, ROI deltas and risk
   scores; `CognitionLayer` turns them into ranking‑weight adjustments.
5. **Ranker refresh** – large changes persist to the metrics DB and retraining
   jobs reload the ranker; failures can trigger embedding backfills.

### Examples

```python
from vector_service.cognition_layer import CognitionLayer
from vector_service.embedding_backfill import schedule_backfill
import asyncio

layer = CognitionLayer()
ctx, sid = layer.query("Improve throughput?")
layer.record_patch_outcome(sid, True)
# contributors gain weight in the ranker

ctx2, sid2 = layer.query("Fix failing tests?")
layer.record_patch_outcome(sid2, False)
asyncio.run(schedule_backfill(dbs=["code"]))
# failure drops weight and refreshes embeddings
```

### Optional dependencies

These modules gracefully degrade when optional packages are missing:

- **ROITracker** – tracks ROI history. Without it, ROI feedback is ignored.
- **UnifiedEventBus** – publishes retrieval and patch events; skipped if absent.
- **VectorMetricsDB** and **PatchHistoryDB** – persist metrics and history; in
  their absence, data is kept in memory only.
- **MenaceMemoryManager** – summarises long texts; falls back to simple
  truncation.
- **SentenceTransformer** – text embeddings for `SharedVectorService`; if
  unavailable, text vectorisation raises `RuntimeError`.
- **tiktoken** – precise token counting for `ContextBuilder`; without it a
  rough estimate is used.
- **UniversalRetriever** – search backend used by `Retriever`; retrieval fails
  if it is missing.

To run without these dependencies simply omit them from the environment; the
service continues with reduced functionality.

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

Additional knobs:

* `RANKER_SCHEDULER_SERVICES` – comma‑separated `module:attr` paths whose
  instances expose `reload_ranker_model`.
* `RANKER_SCHEDULER_VECTOR_DB` / `RANKER_SCHEDULER_PATCH_DB` – override the
  metric and ROI database locations.
* `RANKER_SCHEDULER_MODEL_DIR` – destination directory for newly retrained
  ranker models.

When `RANKER_SCHEDULER_ROI_THRESHOLD` is set, the scheduler also subscribes to
`UnifiedEventBus` on topic `retrieval:feedback` so large ROI swings trigger an
immediate retrain.
