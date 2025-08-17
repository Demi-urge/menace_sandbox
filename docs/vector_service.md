# Vector Service API

The `vector_service.py` module exposes a small service layer used across the
repository.  It wraps low level database helpers with structured logging and
metrics collection so remote agents and development tooling can rely on a
consistent interface.

## Retriever.search

```python
from vector_service import Retriever

r = Retriever()
results = r.search("anomaly in pricing", session_id="agent-42")
for hit in results:
    print(hit["origin_db"], hit["record_id"], hit["score"])
```

`search` performs a semantic lookup via `UniversalRetriever`.  When the primary
retriever returns no results or low scoring matches, the optional
`fallback_retriever` is used and the returned entries are tagged with
`reason="fallback"`.

## ContextBuilder.build

```python
from vector_service import ContextBuilder

builder = ContextBuilder()
context_json = builder.build("refactor payment workflow", session_id="tooling")
```

The wrapper delegates to `context_builder.ContextBuilder` and raises
`MalformedPromptError`, `RateLimitError` or `VectorServiceError` for invalid
input and backend issues.  The returned string is a compact JSON block suitable
for inclusion in language model prompts.

## PatchLogger.track_contributors

```python
from vector_service import PatchLogger

logger = PatchLogger(metrics_db=my_metrics)
logger.track_contributors(["bots:17", "workflow:4"], True, patch_id="abc123")
```

Vector identifiers are split into `(db, id)` pairs and logged either through the
legacy `MetricsDB` or the newer `VectorMetricsDB` interface.

## EmbeddingBackfill.run

```python
from vector_service import EmbeddingBackfill

EmbeddingBackfill().run(session_id="nightly", batch_size=500)
```

`run` discovers all databases derived from `EmbeddableDBMixin` and triggers their
`backfill_embeddings` methods.  It instantiates each database with the configured
backend and processes them sequentially.

## Metrics and logging conventions

Every public method above is decorated with `log_and_time` and `track_metrics`.
Structured log records include the provided `session_id`, allowing correlation
across distributed agents.  Metrics are emitted via the standard Prometheus
exporters.  Future modules should pass meaningful `session_id` values and rely on
these decorators rather than manual timing or logging.
