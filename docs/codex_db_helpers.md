# Codex database helpers

The `codex_db_helpers` module collects training samples from Menace databases
so they can be fed into language‑model prompts. It exposes fetch helpers for
enhancements, workflow summaries, discrepancies and workflow history along with
an `aggregate_samples` convenience wrapper.

## Parameters

All helpers share the following keyword arguments:

- `scope` – visibility of records (`"local"`, `"global"` or `"all"`, default)
  forwarded to `scope_utils.apply_scope_to_query`.
- `sort_by` – column used for ordering; one of `"score"`, `"roi"`,
  `"confidence"` or `"ts"`.
- `limit` – maximum number of rows to return (defaults to `100`).
- `with_embeddings` – attach vector embeddings via `db.vector(id)` when
  available.

The queried tables must expose `id`, a text field (`summary`, `message` or
`details`), and the numeric columns `score`, `roi`, `confidence` and `ts`.

## Example

```python
from codex_db_helpers import aggregate_samples

records = aggregate_samples(
    sources=["enhancement", "workflow_summary"],
    limit_per_source=5,
    sort_by="outcome_score",
    with_vectors=False,
)

prompt = "Examples:\n" + "\n\n".join(
    f"{r['summary']} (ROI={r['roi']:.2f})" for r in records
)
```

## Extending

To add a new data source:

1. Implement a `fetch_<name>` helper that selects `id`, the text column,
   `score`, `roi`, `confidence` and `ts` from the target table while forwarding
   the common parameters.
2. Register the helper in `aggregate_samples` so its output participates
   in the combined ranking.

This keeps the API consistent across data types and preserves scoping and
embedding support.

