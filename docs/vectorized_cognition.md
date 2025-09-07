# Vectorized Cognition

This guide walks through how vector embeddings move through the cognition stack and how feedback steers future retrievals.

## Data Flow

```
vectorizers → EmbeddingBackfill/EmbeddingScheduler → Retriever/ContextBuilder → PatchLogger/ROITracker → ranking model updates
```

1. **Vectorizers** convert raw records for each modality into dense embeddings.
2. **EmbeddingBackfill** populates vectors for new records while **EmbeddingScheduler** keeps them fresh.
3. **Retriever** queries the vector stores and **ContextBuilder** assembles the cognitive context.
4. **PatchLogger** ties retrieved vectors to patches and **ROITracker** records success metrics.
5. Aggregated metrics drive **ranking model updates**, favouring high‑ROI sources on subsequent queries.

## Building Context and Logging Feedback

```python
from vector_service.context_builder import ContextBuilder
from cognition_layer import build_cognitive_context, log_feedback

builder = ContextBuilder("bots.db", "code.db", "errors.db", "workflows.db")

# Build a context for a natural‑language request
context, session_id = build_cognitive_context(
    "optimise cache eviction", top_k=5, context_builder=builder
)

# ...apply patch based on the context...

# Record whether the change succeeded
log_feedback(session_id, True, patch_id="cache-fix-42", context_builder=builder)
```

`build_cognitive_context` wraps `vector_service.cognition_layer.CognitionLayer.query` to return a JSON context blob and session id. `log_feedback` forwards patch outcomes so ROI histories and ranking weights can update.

## Registering New Modalities

Add new record types in `vector_service/registry.py` using `register_vectorizer`:

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

The registry maps each `kind` to its vectorizer and optional database so `EmbeddingBackfill` and the scheduler discover it automatically.

For a full walkthrough including database integration see [modality_registration.md](modality_registration.md).

## ROI Feedback and Database Weights

`ROITracker` aggregates feedback with the originating database. Calling `retrieval_bias()` converts ROI deltas into multiplicative weights that influence ranking:

```python
from roi_tracker import ROITracker

tracker = ROITracker()
tracker.update(0.10, 0.25, retrieval_metrics=[{"origin_db": "bot", "hit": True, "tokens": 12}])
weights = tracker.retrieval_bias()  # e.g. {"bot": 1.2}
```

As feedback accumulates, databases with higher ROI receive larger weights, biasing retrieval toward sources that historically perform better.
