# Context Builder

`ContextBuilder` assembles a compact cross‑database context so language‑model
helpers see only the most relevant history.  It queries the local error, bot,
workflow, enhancement, information, discrepancy and code databases through
[`UniversalRetriever`](universal_retriever.md) and returns a condensed JSON
block that fits within strict token budgets.

## Goals

- Gather related records from multiple sources for a given free‑form query.
- Rank results by vector similarity and ROI or success metrics.
- Emit a minimal structure that downstream tools can slot straight into LLM
  prompts.

## Configuration

```python
from vector_service.context_builder import ContextBuilder

# explicit construction for the standard local databases
builder = ContextBuilder("bots.db", "code.db", "errors.db", "workflows.db")

# configuration with optional tuning
builder = ContextBuilder(
    error_db="errors.db",
    bot_db="bots.db",
    workflow_db="workflows.db",
    code_db="code.db",
    db_weights={"error": 1.5, "code": 0.5},  # optional biasing
    max_tokens=800,  # overall budget
)

context_json = builder.build_context("upload failed", top_k=5, exclude_tags=["failure"])
```

- `top_k` caps how many entries are returned for each origin.
- Scoring weights come from the underlying retriever; tweak parameters on
  `builder.retriever` (for example `link_multiplier`) to emphasise connectivity
  or adjust ranking.  Per-database weights can be supplied via ``db_weights`` or
  configuration to bias towards certain sources.
- `exclude_tags` skips any vectors tagged with the specified values.

`ContextBuilder` can bias ranking toward specific sources and adjust the
overall token budget via optional configuration.  Token usage is estimated for
each candidate segment and the lowest‑scoring entries are iteratively
summarised or removed until the budget is satisfied.  Trimming can optionally
prioritise newer vectors or those with higher ROI:

```yaml
context_builder:
  max_tokens: 800
  db_weights:
    error: 1.5  # emphasise error records
    code: 0.5   # de‑emphasise code snippets
  # during trimming prefer the most recent entries
  # (use "roi" to favour high‑ROI results instead)
  prioritise: newest
```

The defaults keep all databases equally weighted and limit the context to
approximately 800 tokens. Raising a database weight biases ranking toward that
source while lower weights trade recall for diversity.

## Example output

```json
{
  "errors": [{"id": 1, "desc": "...", "metric": 0.8}],
  "bots": [{"id": 2, "desc": "..."}],
  "workflows": [{"id": 3, "desc": "..."}],
  "enhancements": [{"id": 5, "desc": "..."}],
  "information": [{"id": 7, "desc": "..."}],
  "code": [{"id": 6, "desc": "..."}]
}
```

Enhancement entries derive their text from the record's title or description and
use ROI or adoption metrics when available to bias ranking. Information entries
pull from titles or summaries and include any recorded lessons to guide model
prompts.

## Token‑efficiency strategies

- Summaries are shortened via an optional `MenaceMemoryManager`; otherwise a
  small offline helper truncates text.
- Per‑type limits and `max_tokens` prevent runaway context growth; lists are
  trimmed round‑robin until the estimate fits within the budget.
- Optional `prioritise` parameter allows keeping newer or high‑ROI entries
  when trimming.
- Metrics bias the ranking so only high‑value records reach the final JSON.

## Offline operation

All databases are light‑weight SQLite files and the retriever works entirely on
local data, allowing the builder to run without network access.  The fallback
summariser also operates offline, so even without the optional memory manager
`ContextBuilder` produces a fully local context block.

## Integration

`SelfCodingEngine`, `QuickFixEngine`, `BotDevelopmentBot`, `PromptEngine`,
`AutomatedReviewer`, `Watchdog` and `QueryBot` expect a `ContextBuilder` to be
supplied via constructor or method arguments.  Instantiate the builder with the standard
databases and pass it through explicitly:

```python
from vector_service.context_builder import ContextBuilder

builder = ContextBuilder("bots.db", "code.db", "errors.db", "workflows.db")
# e.g. engine = SelfCodingEngine(..., context_builder=builder)
```

## Static enforcement

The repository includes a helper script,
`scripts/check_context_builder_usage.py`, which statically scans Python files
to ensure a ``context_builder`` keyword is threaded through common prompt
helpers.  It flags calls to ``PromptEngine``, ``_build_prompt``,
``generate_patch``, bare ``build_prompt(...)`` helpers and methods named
``build_prompt_with_memory`` when they omit the keyword.  The checker also
inspects direct ``openai.Completion.create`` and ``openai.ChatCompletion.create``
invocations along with the ``chat_completion_create`` wrapper.  To intentionally
skip a call, append ``# nocb`` on the call line (or the line above).  Only direct
``build_prompt(...)`` calls are checked to avoid warning on unrelated methods
with the same name.

```python
import openai

# openai.ChatCompletion.create call is ignored thanks to the marker
openai.ChatCompletion.create([{"role": "user", "content": "hi"}])  # nocb
```

## Custom prompt builders

When rolling your own prompt builders, accept a ``ContextBuilder`` and pass it
through explicitly:

```python
from menace_sandbox.chatgpt_idea_bot import build_prompt
from vector_service.context_builder import ContextBuilder

def my_prompt(client):
    builder = ContextBuilder("bots.db", "code.db", "errors.db", "workflows.db")
    # static check fails if ``context_builder`` is omitted
    return build_prompt(client, context_builder=builder, tags=["ai", "fintech"])
```

