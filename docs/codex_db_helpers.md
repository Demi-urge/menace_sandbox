# Codex database helpers

The `codex_db_helpers` module collects training samples from Menace databases so
they can be fed into language‑model prompts. It exposes dedicated helpers for
enhancements, workflow summaries, discrepancies and workflows along with an
`aggregate_samples` convenience wrapper.

## Functions

- `fetch_enhancements` – summaries from `EnhancementDB`.
- `fetch_summaries` – workflow summaries from `WorkflowSummaryDB`.
- `fetch_discrepancies` – discrepancy messages from `DiscrepancyDB`.
- `fetch_workflows` – saved workflows from `WorkflowDB`.
- `aggregate_samples` – merge and sort samples from all fetchers  
  (`aggregate_examples` is an alias).

For a higher level overview of available sources and sample prompts see
[codex_training_data.md](codex_training_data.md).

## Parameters

All helpers share the following keyword arguments:

- `sort_by` – one of `"confidence"`, `"outcome_score"` or `"timestamp"`; unknown
  values fall back to `"timestamp"`.
- `limit` – maximum number of rows to return (defaults to `100`).
- `include_embeddings` – attach vector embeddings via `db.vector(id)` when
  available.
- `scope` – menace query scope. One of `Scope.LOCAL`, `Scope.GLOBAL` or
  `Scope.ALL` (default).

## Scope

Queries default to `Scope.ALL` via `build_scope_clause`, so records from every
Menace instance participate in fleetwide Codex training. Passing
`scope=Scope.LOCAL` or `scope=Scope.GLOBAL` restricts the results to a single
instance or global templates.

## Example

```python
from codex_db_helpers import aggregate_samples, Scope

records = aggregate_samples(sort_by="timestamp", limit=5, scope=Scope.LOCAL)

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

