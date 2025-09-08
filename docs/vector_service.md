# Vector Service

The `vector_service` module exposes lightweight wrappers around existing
retrieval and logging components. Each wrapper adds structured logging and
Prometheus-style metrics so callers get consistent telemetry without dealing
with individual implementations.

All entry points accept a `session_id` keyword argument which should always be
supplied.  The identifier is propagated to logs and metrics, allowing callers to
correlate related operations across services.

## Deployment bootstrap

Running the environment bootstrap prepares all assets needed for offline
operation. The script fetches `vector_service/minilm/tiny-distilroberta-base.tar.xz`
via `vector_service.download_model` and seeds neutral weights for both
`VectorMetricsDB` and `ROITracker`:

```bash
python scripts/bootstrap_env.py
```

After this step the vector service can start without reaching external
endpoints.

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
from vector_service.context_builder import ContextBuilder

builder = ContextBuilder("bots.db", "code.db", "errors.db", "workflows.db")
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

When patch examples are retrieved, ranking combines vector similarity with the
`enhancement_score` pulled from `PatchHistoryDB`.  The relative influence of the
score can be tuned via ``ContextBuilderConfig.enhancement_weight``.

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

### ROI tags

Patch outcomes may specify an ROI tag describing the quality of a patch. The
available tags are defined by the ``RoiTag`` enum and their effect on ranking
weights is controlled by ``config/roi_tag_outcomes.yaml``. By default
``success`` and ``high-ROI`` increase weights while ``low-ROI`` and
``bug-introduced`` (along with ``needs-review`` and ``blocked``) decrease them.
The YAML file can be edited to customise how each tag influences weight
adjustments. Tags are validated against the enum before any adjustment is
applied.

## CognitionLayer

```python
from vector_service import CognitionLayer
from vector_service.context_builder import ContextBuilder

builder = ContextBuilder("bots.db", "code.db", "errors.db", "workflows.db")
layer = CognitionLayer(context_builder=builder)
ctx, sid = layer.query("What is ROI?")
# ... apply patch ...
layer.record_patch_outcome(sid, True)

# Async
ctx, sid = await layer.query_async("What is ROI?")
await layer.record_patch_outcome_async(sid, True)
```

`CognitionLayer` orchestrates retrieval, context assembly and patch logging.
The `query_async` and `record_patch_outcome_async` helpers mirror their
synchronous counterparts for applications built on `asyncio`.

## EmbeddingBackfill

```python
from vector_service import EmbeddingBackfill

EmbeddingBackfill().run(session_id="bulk")

# Only process WorkflowDB
EmbeddingBackfill().run(session_id="bulk", db="workflows")
```

`EmbeddingBackfill` loads every registered `EmbeddableDBMixin` implementation
from `embedding_registry.json` and invokes `backfill_embeddings` on each,
logging progress for visibility. When a `db` argument is supplied the run is
restricted to matching subclasses. Records flagged by the lightweight license
detector are skipped and logged.

Known databases are listed in `vector_service/embedding_registry.json`. Each
entry maps a short name to the module and class implementing
`EmbeddableDBMixin`:

```json
{
  "bot": {"module": "bot_database", "class": "BotDB"},
  "workflow": {"module": "task_handoff_bot", "class": "WorkflowDB"}
}
```

`EmbeddingBackfill` loads this registry at runtime so new sources can be added
simply by editing the file—no code changes are required.  The JSON object uses
the following format where each key is the canonical name for a database
category and the value specifies the import path and class name of the
`EmbeddableDBMixin` implementation:

```json
{
  "<name>": {"module": "<python module>", "class": "<class name>"}
}
```

To register a new source:

1. Implement a class inheriting from `EmbeddableDBMixin` with an appropriate
   `backfill_embeddings` method.
2. Add an entry to `embedding_registry.json` using the database's short name as
   the key and the module and class where it lives.

Once listed in the registry, the backfill job can process bots, workflows,
enhancements, errors and any future database simply by running the backfill
command.

Metrics emitted by `EmbeddingBackfill.run`:

- `embedding_backfill_runs_total{status="success|failure"}` – counter of
  backfill runs.
- `embedding_backfill_run_duration_seconds` – duration of each run in seconds.
- `embedding_backfill_skipped_total{db,license}` – records dropped due to
  license restrictions.

### Continuous backfills

`EmbeddingBackfill` can run as a long‑lived daemon that watches all registered
`EmbeddableDBMixin` databases for new or updated records:

```bash
python -m vector_service.embedding_backfill --watch --interval 300
```

The example above scans every five minutes.  Alternatively, trigger periodic
backfills with cron:

```
*/30 * * * * python -m vector_service.embedding_backfill >> /var/log/embedding_backfill.log 2>&1
```

Both approaches keep embeddings fresh without manual intervention.

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

