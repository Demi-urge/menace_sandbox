# Context Builder

`ContextBuilder` assembles a compact cross‑database context so language‑model
helpers see only the most relevant history.  It queries the local error, bot,
workflow, discrepancy and code databases through
[`UniversalRetriever`](universal_retriever.md) and returns a condensed JSON
block that fits within strict token budgets.

## Goals

- Gather related records from multiple sources for a given free‑form query.
- Rank results by vector similarity and ROI or success metrics.
- Emit a minimal structure that downstream tools can slot straight into LLM
  prompts.

## Configuration

```python
from menace.context_builder import ContextBuilder

builder = ContextBuilder(
    error_db="errors.db",
    bot_db="bots.db",
    workflow_db="workflows.db",
    code_db="code.db",
)

context_json = builder.build_context("upload failed", top_k=5)
```

- `top_k` caps how many entries are returned for each origin.
- Scoring weights come from the underlying retriever; tweak parameters on
  `builder.retriever` (for example `link_multiplier`) to emphasise connectivity
  or adjust ranking.

`ContextBuilder` can bias ranking toward specific sources via optional
configuration in the application config:

```yaml
context_builder:
  db_weights:
    error: 1.5  # emphasise error records
    code: 0.5   # de‑emphasise code snippets
```

The defaults keep all databases equally weighted.  Raising a database weight
biases ranking toward that source while lower weights trade recall for
diversity.

## Example output

```json
{
  "errors": [{"id": 1, "desc": "...", "metric": 0.8}],
  "bots": [{"id": 2, "desc": "..."}],
  "workflows": [{"id": 3, "desc": "..."}],
  "code": [{"id": 4, "desc": "..."}]
}
```

## Token‑efficiency strategies

- Summaries are shortened via an optional `MenaceMemoryManager`; otherwise a
  small offline helper truncates text.
- Per‑type limits and `max_tokens` prevent runaway context growth.
- Metrics bias the ranking so only high‑value records reach the final JSON.

## Offline operation

All databases are light‑weight SQLite files and the retriever works entirely on
local data, allowing the builder to run without network access.  The fallback
summariser also operates offline, so even without the optional memory manager
`ContextBuilder` produces a fully local context block.

## Integration

`SelfCodingEngine`, `QuickFixEngine`, `BotDevelopmentBot` and
`AutomatedReviewer` instantiate `ContextBuilder` automatically when available to
enrich their prompts with historical context.

