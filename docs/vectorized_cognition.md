# Vectorized Cognition

This document explains how vector-based context flows through the system and how feedback updates influence future retrieval.

## Data Flow

```
vectorizers → EmbeddingBackfill / EmbeddingScheduler → Retriever / ContextBuilder → PatchLogger / ROITracker → ranking model updates
```

1. **Vectorizers** turn records of each modality into embedding vectors.
2. **EmbeddingBackfill** populates vector stores for new records.  **EmbeddingScheduler** periodically refreshes them.
3. **Retriever** searches the stores and **ContextBuilder** assembles a cognitive context from the top results.
4. **PatchLogger** and **ROITracker** capture which vectors affected a patch and the resulting return on investment (ROI).
5. The accumulated metrics drive **ranking model updates**, favouring high‑ROI sources on subsequent queries.

## Building Context and Logging Feedback

```python
from cognition_layer import build_cognitive_context, log_feedback

# Build a context for a natural‑language request
context, session_id = build_cognitive_context("optimise cache eviction", top_k=5)

# ...apply patch based on the context...

# Record whether the change succeeded
log_feedback(session_id, True, patch_id="cache-fix-42")
```

`build_cognitive_context` wraps the vector service to return a JSON context blob and a session id.  `log_feedback` forwards patch outcomes so metrics and ROI histories can be updated.

## Registering New Modalities

New record types are registered in `vector_service/registry.py` via `register_vectorizer`:

```python
from vector_service.registry import register_vectorizer

register_vectorizer(
    "sentiment",
    "sentiment_vectorizer",
    "SentimentVectorizer",
    db_module="sentiment_db",
    db_class="SentimentDB",
)
```

The registry maps the `kind` to both the vectorizer and, optionally, its database.  `EmbeddingBackfill` and the scheduler use this mapping to discover new modalities automatically.

## ROI Feedback and Database Weights

`ROITracker` aggregates feedback with the originating database.  Calling `retrieval_bias()` converts ROI deltas into multiplicative weights used by `ContextBuilder`, so databases with higher ROI are ranked earlier.

```python
from roi_tracker import ROITracker

tracker = ROITracker()
tracker.update(0.10, 0.25, retrieval_metrics=[{"origin_db": "bot", "hit": True, "tokens": 12}])
weights = tracker.retrieval_bias()  # e.g. {"bot": 1.2}
```

These weights adapt ranking as feedback accumulates, gradually biasing retrieval toward databases that historically produce better outcomes.
