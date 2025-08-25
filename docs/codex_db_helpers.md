# Codex database helpers

The `codex_db_helpers` module collects training samples from Menace databases so
they can be fed into language‑model prompts. It exposes fetch helpers for
enhancements, workflow summaries, discrepancies and workflows along with an
`aggregate_samples` convenience wrapper.

For a higher level overview of available sources and sample prompts see
[codex_training_data.md](codex_training_data.md).

## Parameters

All helpers share the following keyword arguments:

- `sort_by` – one of `"confidence"`, `"outcome_score"` or `"timestamp"`.
- `limit` – maximum number of rows to return (defaults to `100`).
- `include_embeddings` – attach vector embeddings via `db.vector(id)` when
  available.

Queries use `Scope.ALL` so records from all menace instances are returned.

## Example

```python
from codex_db_helpers import aggregate_samples

records = aggregate_samples(sort_by="timestamp", limit=5)

prompt = "Examples:\n" + "\n\n".join(
    f"{r.source}: {r.content}" for r in records
)
```

## Extending

To add a new data source:

1. Implement a `fetch_<name>` helper that returns a list of
   :class:`TrainingSample` objects and accepts the standard parameters.
2. Register the helper in `aggregate_samples` so its output participates in the
   combined ranking.

This keeps the API consistent across data types and preserves embedding
support.

