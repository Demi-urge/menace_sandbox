# Vector Service

The `vector_service` module exposes lightweight wrappers around existing
retrieval and logging components. Each wrapper adds structured logging and
Prometheus-style metrics so callers get consistent telemetry without dealing
with individual implementations.

All entry points accept a `session_id` keyword argument which should always be
supplied.  The identifier is propagated to logs and metrics, allowing callers to
correlate related operations across services.

## Retriever

```python
from vector_service import Retriever

r = Retriever()
results = r.search("upload failed", session_id="abc123")
```

`Retriever` delegates to `universal_retriever.UniversalRetriever` and returns
serialisable dictionaries. Log entries include latency, result size and an
optional `session_id` for tracing queries.

## ContextBuilder

```python
from vector_service import ContextBuilder

builder = ContextBuilder()
context = builder.build("fix failing tests")
```

This helper summarises related bots, workflows and errors while recording
metrics for each build.

## PatchLogger

```python
from vector_service import PatchLogger

logger = PatchLogger()
logger.track_contributors(["bot:1", "workflow:2"], True,
                          patch_id="42", session_id="abc123")
```

Vector identifiers may optionally specify the origin database using the
`"origin:id"` format. Outcomes are forwarded to `data_bot.MetricsDB` when
available or to `vector_metrics_db.VectorMetricsDB` as a fallback.

Metrics emitted by `PatchLogger.track_contributors`:

- `patch_logger_track_contributors_total{status="success|failure|error"}` –
  counter of contributor tracking attempts.
- `patch_logger_track_contributors_duration_seconds` – duration of each call.

## EmbeddingBackfill

```python
from vector_service import EmbeddingBackfill

EmbeddingBackfill().run(session_id="bulk")
```

`EmbeddingBackfill` discovers every `EmbeddableDBMixin` subclass and invokes
`backfill_embeddings` on each, logging progress for visibility.

Metrics emitted by `EmbeddingBackfill.run`:

- `embedding_backfill_runs_total{status="success|failure"}` – counter of
  backfill runs.
- `embedding_backfill_run_duration_seconds` – duration of each run in seconds.

