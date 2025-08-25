# Codex training data

The `codex_db_helpers` module unifies access to textual training samples stored across Menace databases. It exposes fetch helpers for multiple sources and an `aggregate_samples` wrapper that merges them into a single ordered list.

## Data sources

These sources are available out of the box:

- **enhancement** – summaries and outcome scores from `EnhancementDB`.
- **workflow_summary** – short workflow descriptions from `WorkflowSummaryDB`.
- **discrepancy** – discrepancy reports with confidence scores from `DiscrepancyDB`.
- **workflow** – stored workflows from `task_handoff_bot.WorkflowDB`.

## Sorting, scope and embeddings

All helpers accept:

- `sort_by` – `"confidence"`, `"outcome_score"` or `"timestamp"` determine ordering.
- `limit` – number of rows to fetch per source.
- `include_embeddings` – when `True`, attach embedding vectors via the respective database's `vector(id)` API.
- `scope` – restrict records to a menace instance. Accepts `Scope` values or
  `"local"`, `"global"` or `"all"` (default).

## Example

```python
from codex_db_helpers import aggregate_samples, Scope
import openai

samples = aggregate_samples(sort_by="outcome_score", limit=10, scope=Scope.LOCAL)

prompt = "Examples:\n" + "\n\n".join(
    f"{s.source}: {s.content} (score={s.outcome_score})" for s in samples
)

completion = openai.Completion.create(
    model="code-davinci-002",
    prompt=prompt,
    max_tokens=200,
)
print(completion["choices"][0]["text"])
```

The snippet aggregates recent records, formats them as a prompt and sends it to a Codex model for completion.

For an API reference and extension guidelines see [codex_db_helpers.md](codex_db_helpers.md).
