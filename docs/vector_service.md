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
serialisable dictionaries. Results are always filtered to drop hits flagged by
the license detector and to attach semantic risk annotations. This filtering is
mandatory and cannot be disabled. Log entries include latency, result size and
an optional `session_id` for tracing queries.

## ContextBuilder

```python
from vector_service import ContextBuilder

builder = ContextBuilder()
context = builder.build("fix failing tests")
```

This helper summarises related bots, workflows and errors while recording
metrics for each build.  Each entry now exposes reliability metrics pulled from
`VectorMetricsDB` and patch‑safety flags when available:

```json
{
  "id": 1,
  "desc": "alpha bot",
  "metric": 10.0,
  "win_rate": 0.8,
  "regret_rate": 0.1,
  "flags": {"license": "GPL", "semantic_alerts": ["unsafe"]}
}
```

To inspect these fields without inflating the prompt context, pass
`return_metadata=True`:

```python
context, meta = builder.build_context("fix failing tests", return_metadata=True)
```

`context` contains the compact summaries while `meta` mirrors the structure with
the additional `win_rate`, `regret_rate` and `flags` details for each entry.

## PatchLogger

```python
from vector_service import PatchLogger

logger = PatchLogger()
logger.track_contributors(["bot:1", "workflow:2"], True,
                          patch_id="42", session_id="abc123",
                          contribution=0.5)
```

Vector identifiers may optionally specify the origin database using the
`"origin:id"` format. Outcomes are forwarded to `data_bot.MetricsDB` when
available or to `vector_metrics_db.VectorMetricsDB` as a fallback.  An optional
`contribution` parameter forwards a weighting to these databases.

Metrics emitted by `PatchLogger.track_contributors`:

- `patch_logger_track_contributors_total{status="success|failure|error"}` –
  counter of contributor tracking attempts.
- `patch_logger_track_contributors_duration_seconds` – duration of each call.

## EmbeddingBackfill

```python
from vector_service import EmbeddingBackfill

EmbeddingBackfill().run(session_id="bulk")

# Only process WorkflowDB
EmbeddingBackfill().run(session_id="bulk", db="workflows")
```

`EmbeddingBackfill` discovers every `EmbeddableDBMixin` subclass and invokes
`backfill_embeddings` on each, logging progress for visibility. When a `db`
argument is supplied the run is restricted to matching subclasses. Records
flagged by the lightweight license detector are skipped and logged.

Metrics emitted by `EmbeddingBackfill.run`:

- `embedding_backfill_runs_total{status="success|failure"}` – counter of
  backfill runs.
- `embedding_backfill_run_duration_seconds` – duration of each run in seconds.

## HTTP API configuration

The accompanying `vector_service_api` module exposes these helpers through a
FastAPI application.  Basic authentication and rate limiting can be configured
via environment variables:

- `VECTOR_SERVICE_API_TOKEN` – if set, clients must supply the same value in the
  `X-API-Token` header (or `token` query parameter).
- `VECTOR_SERVICE_RATE_LIMIT` – maximum number of requests allowed per
  `VECTOR_SERVICE_RATE_WINDOW` seconds for each API token or source IP.  Both
  variables default to `60` seconds and `60` requests respectively.

Example usage:

```bash
export VECTOR_SERVICE_API_TOKEN="secret"
export VECTOR_SERVICE_RATE_LIMIT=30
uvicorn vector_service_api:app

# Authenticated request
curl -H "X-API-Token: secret" -d '{"query": "upload failed"}' \
    http://localhost:8000/search
```

