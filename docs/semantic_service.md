# Semantic Service (deprecated)

The old `semantic_service` module has been renamed to `vector_service`.
Update imports accordingly:

```python
from vector_service import Retriever, ContextBuilder, PatchLogger, EmbeddingBackfill
```

An HTTP layer in `vector_service_api.py` exposes `/search`, `/build-context`,
`/track-contributors` and `/backfill-embeddings` endpoints for these helpers.
See [vector_service.md](vector_service.md) and
[vector_service_api.md](vector_service_api.md) for usage examples.
