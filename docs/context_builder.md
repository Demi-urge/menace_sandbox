# Context Builder

`ContextBuilder` assembles a compact cross‑database context so language‑model
helpers see only the most relevant history.  It queries the local error, bot,
workflow, enhancement, information, discrepancy and code databases through
[`UniversalRetriever`](universal_retriever.md) and returns a condensed JSON
block that fits within strict token budgets.  Typical usage wires in
`bots.db`, `code.db`, `errors.db` and `workflows.db` and compresses retrieved
snippets before embedding them in prompts.

When used via :class:`SelfCodingManager` the builder is refreshed automatically
before every patch.  The manager calls ``refresh_db_weights`` to ensure scoring
weights are up to date and logs each retrieval ``session_id`` to
``PatchHistoryDB`` so subsequent metrics can be correlated with the context
that produced them.

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

## Building prompts

``ContextBuilder.build_prompt`` converts a user intent into a ``Prompt``
dataclass ready for downstream LLM calls. The method optionally expands the
intent into latent queries and ingests error logs or extra metadata. Retrieved
snippets are compressed, deduplicated and ranked by combining vector similarity,
ROI, recency and safety signals. The highest scoring examples are packed into
the prompt while respecting the configured token budget.

```python
from vector_service.context_builder import ContextBuilder, build_prompt

builder = ContextBuilder()
prompt = builder.build_prompt(
    "optimise database",
    intent={"ticket": 123, "intent_tags": ["perf"]},
    error_log="timeout",
    latent_queries=["speed up queries"],
)
print(prompt.metadata["intent"])
print(prompt.metadata["vectors"][:1])

# or use the module level helper
prompt2 = build_prompt("optimise database")
```

Example output::

```
{'ticket': 123, 'intent_tags': ['perf']}
[('errors:17', 0.8)]
```

Intent metadata and retrieved vector identifiers are merged into the returned
prompt's ``metadata`` so downstream components can reason about both the user's
intent and the surrounding context.

Avoid constructing prompt strings inline; always call ``build_prompt`` and run
``python scripts/check_context_builder_usage.py`` to statically flag any missing
``context_builder`` wiring.  Results from ``context_builder.build(...)`` must be
passed directly to the client; concatenating them with other strings or lists is
reported, and direct ``ask_with_memory`` calls are discouraged.

Configuration knobs controlling this method live under ``context_builder`` in the
main settings file:

```yaml
context_builder:
  prompt_max_tokens: 400       # token budget for build_prompt
  prompt_score_weight: 1.2     # bias similarity vs ROI when ranking examples
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
- Retrieved code snippets are compressed with `snippet_compressor` before
  being embedded in prompts.

## Offline operation

All databases are light‑weight SQLite files and the retriever works entirely on
local data, allowing the builder to run without network access.  The fallback
summariser also operates offline, so even without the optional memory manager
`ContextBuilder` produces a fully local context block.

## Integration

`SelfCodingEngine`, `QuickFixEngine`, `BotDevelopmentBot`, `PromptEngine`,
`AutomatedReviewer`, `Watchdog` and `QueryBot` expect a `ContextBuilder` to be
supplied via constructor or method arguments. Helpers such as `_build_prompt`,
`build_prompt` and `generate_patch` must receive this keyword or they will raise
a `ValueError` or fall back to empty prompts. Instantiate the builder with the
standard databases and pass it through explicitly:

```python
from vector_service.context_builder import ContextBuilder

builder = ContextBuilder("bots.db", "code.db", "errors.db", "workflows.db")
```

### Bot usage examples

#### BotDevelopmentBot

```python
from bot_development_bot import BotDevelopmentBot, BotSpec
from self_coding_engine import SelfCodingEngine
from menace_memory_manager import MenaceMemoryManager

memory_mgr = MenaceMemoryManager()
engine = SelfCodingEngine("code.db", memory_mgr, context_builder=builder)
bot = BotDevelopmentBot(context_builder=builder, engine=engine)
bot._build_prompt(BotSpec(name="demo", purpose="test"), context_builder=builder)
```

#### AutomatedReviewer

```python
from automated_reviewer import AutomatedReviewer

reviewer = AutomatedReviewer(context_builder=builder)
reviewer.handle({"bot_id": "1", "severity": "critical"})
```

#### QuickFixEngine

```python
from quick_fix_engine import QuickFixEngine, generate_patch
from error_bot import ErrorDB
from self_coding_engine import SelfCodingEngine
from model_automation_pipeline import ModelAutomationPipeline
from data_bot import DataBot
from bot_registry import BotRegistry
from self_coding_manager import SelfCodingManager

manager = SelfCodingManager(
    SelfCodingEngine(),
    ModelAutomationPipeline(),
    data_bot=DataBot(),
    bot_registry=BotRegistry(),
)
engine = QuickFixEngine(ErrorDB(), manager, context_builder=builder)
generate_patch("sandbox_runner", context_builder=builder)
```

## Failure modes

Leaving out the builder causes helper methods that require contextual snippets
to either throw a `ValueError("ContextBuilder is required")` or silently fall
back to empty prompts. The latter severely degrades code generation quality
because no historical data is supplied to the model.

## Troubleshooting

If you encounter `ContextBuilder validation failed` in logs, verify that
`bots.db`, `code.db`, `errors.db` and `workflows.db` exist and are readable.
Running `builder.validate()` or `builder.refresh_db_weights()` can surface
missing tables or corrupted files.

## Static enforcement

The repository includes a helper script,
`scripts/check_context_builder_usage.py`, which statically scans Python files
to ensure a ``context_builder`` keyword is threaded through common prompt
helpers.  It flags calls to ``PromptEngine``, ``_build_prompt``,
``generate_patch``, bare ``build_prompt(...)`` helpers and methods named
``build_prompt_with_memory`` when they omit the keyword. The checker also
inspects any direct chat-completion or other remote LLM calls so that
generation consistently routes through ``SelfCodingEngine``. To intentionally
skip a call, append ``# nocb`` on the call line (or the line above). Only
direct ``build_prompt(...)`` calls are checked to avoid warning on unrelated
methods with the same name. Variables assigned from ``LLMClient``-like classes
are tracked so that subsequent ``instance.generate(...)`` invocations require
the keyword as well; aliases such as ``llm`` or ``model`` are heuristically
checked even without a prior assignment.

Functions that invoke ``ContextBuilder.build`` must accept an explicit builder
argument.  A parameter defaulting to ``None`` or falling back to a new builder
inside the function body triggers an error.  This prevents helpers from silently
creating builders when callers forget to provide one.

```python
def demo(builder=None):  # flagged
    builder.build()

def ok(builder):
    builder.build()  # passes
```

```python
from self_coding_engine import SelfCodingEngine

# SelfCodingEngine.generate call is ignored thanks to the marker
SelfCodingEngine().generate([{"role": "user", "content": "hi"}])  # nocb
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

