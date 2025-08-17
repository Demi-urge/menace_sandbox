# Semantic Service (deprecated)

The legacy `semantic_service` module has been replaced by
[`vector_service`](vector_service.md). Update imports accordingly:

```python
from vector_service import Retriever, ContextBuilder, PatchLogger, EmbeddingBackfill
```

An HTTP layer in `vector_service_api.py` exposes `/search`, `/build-context`,
`/track-contributors` and `/backfill-embeddings` endpoints for these helpers.

## Migration from semantic_service

1. **Replace imports**  
   `from semantic_service import X` → `from vector_service import X`
2. **Update configuration**  
   Rename any environment variables or settings referencing `semantic_service`.
3. **Pass `session_id`**  
   All helpers accept a `session_id`; omitting it makes tracing harder.
4. **Avoid direct database access**  
   Use the service layer instead of interacting with embedding stores yourself.

Common pitfalls:

- Mixing old and new imports within the same module.
- Forgetting to adjust tests or deployment scripts.
- Missing `session_id` on long-running processes.

## Typical usage

### Bots

```python
from vector_service import Retriever, ContextBuilder

def plan_work(query):
    r = Retriever()
    builder = ContextBuilder()
    results = r.search(query, session_id="bot-42")
    context = builder.build(query, session_id="bot-42")
    return results, context
```

### Error monitors

```python
from vector_service import Retriever, PatchLogger

def analyse_error(message):
    hits = Retriever().search(message, session_id="monitor-1")
    PatchLogger().track_contributors(
        [f"error:{h['id']}" for h in hits], False,
        patch_id="auto", session_id="monitor-1")
    return hits
```

See [`vector_service.md`](vector_service.md) and
[`vector_service_api.md`](vector_service_api.md) for complete API details.

