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
from vector_service.context_builder import ContextBuilder
from vector_service.cognition_layer import CognitionLayer

tracker = ROITracker()
builder = ContextBuilder()
layer = CognitionLayer(context_builder=builder, roi_tracker=tracker)

ctx, session_id = layer.query("How can I fix latency?")
# ...apply patch based on ctx...
layer.record_patch_outcome(session_id, True, contribution=1.0)
# ranking weights now reflect ROI feedback
```

The final call updates ROI metrics and adjusts ranking weights.

## Patch example lookup

Modules can retrieve stored code patch examples with a single call to
``search_patches``.  The helper initialises a ``PatchRetriever`` using the
backend and metric from ``vector_store`` configuration.  Scores are normalised
to the ``[0, 1]`` range regardless of backend:

```python
from vector_service.retriever import search_patches

examples = search_patches("fix bug", top_k=3)
```

The returned list contains ``origin_db``, ``record_id`` and ``score`` fields
similar to results from ``Retriever.search``.

When available, the retriever looks up each patch's ``enhancement_score`` in
``PatchHistoryDB`` and boosts the similarity score.  The balance between
similarity and enhancement can be tuned with
``ContextBuilderConfig.enhancement_weight``.

## Adding new modalities

The vector service dynamically discovers vectorisers.  Any module placed under
``vector_service`` whose name ends in ``_vectorizer`` and exposes a class ending
with ``Vectorizer`` is registered automatically.  The registry key is derived
from the class name (``ActionVectorizer`` → ``"action"``).

To enable embedding backfills for a new modality, set ``DB_MODULE`` and
``DB_CLASS`` attributes on the module or the class to point at the corresponding
database implementation.  Once the module is added no source code changes are
required—the new kind becomes available to both the vectoriser and
``embedding_backfill``.

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
from vector_service.context_builder import ContextBuilder
from vector_service.embedding_backfill import schedule_backfill
import asyncio

layer = CognitionLayer(context_builder=ContextBuilder())
ctx, sid = layer.query("Improve throughput?")
layer.record_patch_outcome(sid, True)
# contributors gain weight in the ranker

ctx2, sid2 = layer.query("Fix failing tests?")
layer.record_patch_outcome(sid2, False)
asyncio.run(schedule_backfill(dbs=["code"]))
# failure drops weight and refreshes embeddings
```

## Session ROI analytics

`analytics/session_roi.py` correlates retrieval sessions stored in
`VectorMetricsDB` with their patch outcomes. It can group ROI metrics by
origin type for **bots**, **workflows**, **enhancements** and **errors**:

```bash
python -m analytics.session_roi --db vector_metrics.db --by-type
```

Sample output::

```json
{
  "bots": {"bots": {"success_rate": 1.0, "roi_delta": 0.5}},
  "workflows": {},
  "enhancements": {},
  "errors": {}
}
```

The mapping highlights which databases yield successful patches and how much
value they contribute. The same summary is accessible via
`VectorMetricsAggregator.origin_stats`, and programmatically through
`CognitionLayer.roi_stats()` for external dashboards. Running the aggregator
also writes `vector_origin_stats.json`.

## Failure embeddings and risk penalties

`PatchLogger` keeps embeddings of vectors that led to failed patches. When a
patch outcome is negative, the contributing vectors' metadata is fed into
`PatchSafety.record_failure`. Later retrievals call `PatchSafety.evaluate`, which
computes a similarity score against the stored failures. High similarity raises
the risk score returned by `PatchLogger`, allowing `CognitionLayer` to demote
those origins in the ranker.

```python
from vector_service.patch_logger import PatchLogger

logger = PatchLogger()
meta = {"error": {"message": "timeout"}}

# failing patch: record its embedding
logger.track_contributors({"code:1": 1.0}, False, retrieval_metadata={"code:1": meta})

# later retrieval: similar vector receives a penalty
scores = logger.track_contributors({"code:2": 1.0}, True, retrieval_metadata={"code:2": meta})
print(scores["code"])  # non-zero similarity -> down-ranked
```

### Optional dependencies

These modules gracefully degrade when optional packages are missing:

- **ROITracker** – tracks ROI history. Without it, ROI feedback is ignored.
- **UnifiedEventBus** – publishes retrieval and patch events; skipped if absent.
- **VectorMetricsDB** and **PatchHistoryDB** – persist metrics and history; in
  their absence, data is kept in memory only.
- **MenaceMemoryManager** – summarises long texts; falls back to simple
  truncation.
- **SentenceTransformer** – text embeddings for `SharedVectorService`. If the
  package is missing, the service falls back to a bundled DistilRoBERTa model.
- **tiktoken** – precise token counting for `ContextBuilder`; without it a
  rough estimate is used.
- **UniversalRetriever** – search backend used by `Retriever`; retrieval fails
  if it is missing.

To run without these dependencies simply omit them from the environment; the
service continues with reduced functionality.

### Local embedding model

Text embeddings can fall back to a tiny DistilRoBERTa encoder expected at
`vector_service/minilm/tiny-distilroberta-base.tar.xz`. The repository omits
this binary archive; download it manually with:

```bash
python -m vector_service.download_model
```

The command fetches the model from Hugging Face and writes the compressed
archive to the expected location.

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
environment variables and invoke the module to enable it or rely on
`vector_service_api.py` which calls `start_scheduler_from_env` during startup
once initialised via `create_app(ContextBuilder(...))`.

```bash
export RANKER_SCHEDULER_INTERVAL=3600  # seconds between retrains (0 disables)
export RANKER_SCHEDULER_ROI_THRESHOLD=5  # optional ROI delta for immediate retrain
export RANKER_SCHEDULER_RISK_THRESHOLD=2  # optional risk delta for immediate retrain
export RANKER_SCHEDULER_EVENT_LOG=events.db  # optional SQLite log for bus events
export RANKER_SCHEDULER_RABBITMQ_HOST=localhost  # optional RabbitMQ host
python -m analytics.ranker_scheduler
```

With these variables present the FastAPI app automatically enables the scheduler
and listens for ROI and risk feedback so large swings trigger immediate retrains.

Additional knobs:

* `RANKER_SCHEDULER_SERVICES` – comma‑separated `module:attr` paths whose
  instances expose `reload_ranker_model`.
* `RANKER_SCHEDULER_VECTOR_DB` / `RANKER_SCHEDULER_PATCH_DB` – override the
  metric and ROI database locations.
* `RANKER_SCHEDULER_MODEL_DIR` – destination directory for newly retrained
  ranker models.
* `RANKER_SCHEDULER_EVENT_LOG` – persist `UnifiedEventBus` events to this SQLite file.
* `RANKER_SCHEDULER_RABBITMQ_HOST` – mirror bus events to RabbitMQ at this host.

When `RANKER_SCHEDULER_ROI_THRESHOLD` or `RANKER_SCHEDULER_RISK_THRESHOLD` is
set, the scheduler subscribes to `UnifiedEventBus` on topic `roi:update` and
accumulates ROI and risk deltas per origin so large swings trigger an immediate
retrain.

## Embedding backfill daemon

Trigger a one-off backfill of the core databases using the Menace CLI:

```bash
python menace_cli.py embed core
```

Progress for each database is displayed and any skipped records or licensing
violations are surfaced. `EmbeddingBackfill` also exposes a lightweight daemon
that watches registered databases for new or modified records.  Invoke it as a
module with the `--watch` flag to keep embeddings synchronised:

```bash
python -m vector_service.embedding_backfill --watch --interval 30
```

The daemon polls every `--interval` seconds (default 60) and uses
`SharedVectorService.vectorise_and_store` to persist embeddings for any unseen
records.  Provide `--db NAME` multiple times to restrict the watch list.

A systemd unit file [`systemd/embedding_backfill_watcher.service`](../systemd/embedding_backfill_watcher.service)
runs the daemon in the background on production hosts.
