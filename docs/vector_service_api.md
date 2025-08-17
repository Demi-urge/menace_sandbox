# Vector Service HTTP API

The `vector_service_api` module exposes a tiny [FastAPI](https://fastapi.tiangolo.com/)
application that wraps helpers from [`semantic_service`](semantic_service.md).
Each endpoint returns a JSON object with a `status` field and timing metrics.

## `POST /search`
Run semantic search through `semantic_service.Retriever`.

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
Generate a contextual string using `semantic_service.ContextBuilder`.

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
Record contributor outcomes via `semantic_service.PatchLogger`.

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
Trigger an embedding backfill through `semantic_service.EmbeddingBackfill`.

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
