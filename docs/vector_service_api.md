# Vector Service HTTP API

The `vector_service_api` module exposes a tiny [FastAPI](https://fastapi.tiangolo.com/)
application that wraps helpers from [`vector_service`](vector_service.md).
Each endpoint returns a JSON object with a `status` field and timing metrics.
Before sending requests, initialise the app with an explicit
`ContextBuilder`.  The builder and related services are stored on
`app.state` so handlers can access them via `request.app.state`:

```python
from context_builder_util import create_context_builder
import vector_service_api

vector_service_api.create_app(create_context_builder())
```

## `POST /search`
Run semantic search through `vector_service.Retriever`.

```bash
curl -X POST http://localhost:8000/search \
     -H 'Content-Type: application/json' \
     -d '{"query": "upload failed", "top_k": 3}'
```

Response:
```json
{
  "status": "ok",
  "data": [/* search results */],
  "metrics": {"duration": 0.12, "result_size": 1}
}
```

## `POST /build-context`
Generate a contextual string using `vector_service.ContextBuilder`.

```bash
curl -X POST http://localhost:8000/build-context \
     -H 'Content-Type: application/json' \
     -d '{"task_description": "fix failing tests"}'
```

Response:
```json
{
  "status": "ok",
  "data": "context string",
  "metrics": {"duration": 0.05, "result_size": 42}
}
```

## `POST /track-contributors`
Record contributor outcomes via `vector_service.PatchLogger`.

```bash
curl -X POST http://localhost:8000/track-contributors \
     -H 'Content-Type: application/json' \
     -d '{"vector_ids": ["bot:1"], "result": true, "patch_id": "abc"}'
```

Response:
```json
{
  "status": "ok",
  "metrics": {"duration": 0.01}
}
```

## `POST /backfill-embeddings`
Trigger an embedding backfill through `vector_service.EmbeddingBackfill`.

```bash
curl -X POST http://localhost:8000/backfill-embeddings \
     -H 'Content-Type: application/json' \
     -d '{"batch_size": 100}'
```

Response:
```json
{
  "status": "ok",
  "metrics": {"duration": 0.2}
}
```

All endpoints return HTTP 500 with a `detail` message if the underlying
service raises a `VectorServiceError`.

## Vector Database Service

`vector_service.vector_database_service` provides a lightweight daemon for
embedding records and querying stored vectors. Start it with:

```bash
python -m vector_service
```

The service exposes three endpoints:

* `POST /add` – vectorises a record and stores the embedding.
* `POST /query` – returns the nearest neighbours for a supplied record.
* `GET /status` – simple health check.

Bots should call the `GET /status` endpoint and ensure the daemon is running
before invoking `ContextBuilder` or issuing database queries.
