# GPT Memory

`gpt_memory.py` provides lightweight persistence for GPT interactions.  It can
store prompts and responses with tags, optionally embed each prompt for semantic
search and compact old records into concise summaries.

## Tag taxonomy

To keep memory queries consistent across modules, GPT interactions use a small
set of canonical tags:

- `feedback` – captures the outcome or evaluation of an action.
- `improvement_path` – records planned follow-up work or patch identifiers.
- `error_fix` – marks code changes or suggestions intended to address faults.
- `insight` – stores ideas, predictions, or general observations.

These tags are defined in `log_tags.py` and mirrored by `gpt_memory.STANDARD_TAGS`
so new modules can reference the same taxonomy when logging.

Every call to `gpt_client.ask` should pass a `tags` list using these constants to
describe the intent of the interaction. Use `ERROR_FIX` for patches and bug
resolutions, `INSIGHT` when brainstorming or generating ideas, and
`FEEDBACK`/`IMPROVEMENT_PATH` to capture evaluation results and follow-up work.
This keeps memory queries and future contributions consistent with the
definitions in `log_tags.py`.

## LocalKnowledgeModule

`LocalKnowledgeModule` combines a `GPTMemoryManager` with the
`GPTKnowledgeService` summariser.  It exposes a tiny facade for logging new
entries, generating insights and building context.  By default it stores data in
`gpt_memory.db` and can be shared across modules using
`local_knowledge_module.init_local_knowledge`.  Reusing the same database path
allows memories and generated insights to persist between separate runs of the
program.

### Persistence across sessions

When initialised with a `db_path` that already exists, the module reloads prior
interactions so subsequent prompts gain access to earlier feedback, fixes and
improvement plans.  The path can also be supplied via the `GPT_MEMORY_DB`
environment variable:

```bash
export GPT_MEMORY_DB="persistent.db"
```

## Removed helper `ask_with_memory`

The helper ``memory_aware_gpt_client.ask_with_memory`` has been removed. Build
prompts explicitly via :meth:`ContextBuilder.build_prompt` and pass the result
to your client via :meth:`LLMClient.generate`:

```python
from vector_service.context_builder import ContextBuilder
from log_tags import FEEDBACK

prompt = context_builder.build_prompt(
    "How did it go?",
    intent_metadata={"user_query": "How did it go?"},
)
resp = client.generate(
    prompt,
    context_builder=context_builder,
    tags=[FEEDBACK],
)
```

References to ``ask_with_memory`` are flagged by
``scripts/check_context_builder_usage.py``. Tags should use the canonical labels
from ``log_tags.py`` (``FEEDBACK``, ``IMPROVEMENT_PATH``, ``ERROR_FIX``,
``INSIGHT``) so other tools can query the history consistently.

## Configuration

Several environment variables influence how memory behaves:

- **`GPT_MEMORY_DB`** – path to the SQLite database used by
  `LocalKnowledgeModule` (defaults to `gpt_memory.db`).
- **`GPT_MEMORY_RETENTION`** – comma separated `tag=count` pairs limiting how
  many entries of each type are kept when the store is compacted.
- **`GPT_AUTO_REFRESH_INSIGHTS`** – when set to `0`/`false`, disables the
  background refresh of summarised insights triggered after each log.

## Unified interface

All memory backends now implement `GPTMemoryInterface` providing four core
operations: `store`, `retrieve`, `log_interaction` and `search_context`.  The
SQLite manager, the shared `MenaceMemoryManager` and `MemoryBot` all adhere to
this protocol so higher level components can depend on the interface rather than
concrete implementations.

## GPTMemoryManager

```python
from gpt_memory import GPTMemoryManager
from unified_event_bus import UnifiedEventBus

bus = UnifiedEventBus()
mgr = GPTMemoryManager("memory.db", event_bus=bus)
mgr.log_interaction("user question", "assistant reply", tags=["note"])
entries = mgr.search_context("question")
```

Key configuration options:

- `db_path` – SQLite file used for storage (`"gpt_memory.db"` by default).
- `embedder` – optional `SentenceTransformer` model enabling semantic search.
- `event_bus` – optional `UnifiedEventBus` publishing a `"memory:new"` event for
  each logged interaction.

### Logging with tags

Use `log_interaction` to record prompts and responses. Tags should come from
the constants in `log_tags.py` so downstream tools can query them
consistently:

```python
from log_tags import ERROR_FIX

mgr.log_interaction("bug report", "fixed", tags=[ERROR_FIX])
```

When an event bus is supplied every call also publishes a `memory:new` event.
The payload contains the original `prompt` and list of `tags` so consumers can
filter or enrich the entry before persisting it elsewhere.  Other services may
subscribe to react to new memories in real time.

### Knowledge graph ingestion

`KnowledgeGraph` can subscribe to the event bus and ingest entries
automatically:

```python
from knowledge_graph import KnowledgeGraph

kg = KnowledgeGraph("kg.gpickle")
kg.listen_to_memory(bus, mgr)  # ingests on each memory:new event
# or import the current history manually
kg.ingest_gpt_memory(mgr)
```

### Sample configuration for persistent learning

The snippet below wires a persistent `GPTMemoryManager` and `KnowledgeGraph`
through a shared `UnifiedEventBus`.  Each call to `log_interaction` emits a
`memory:new` event which the graph listens for, allowing cross-session learning
as both the SQLite database and graph file are reused across runs:

```python
from unified_event_bus import UnifiedEventBus
from gpt_memory import GPTMemoryManager
from knowledge_graph import KnowledgeGraph

bus = UnifiedEventBus()
memory = GPTMemoryManager("persistent.db", event_bus=bus)
graph = KnowledgeGraph("kg.gpickle")
graph.listen_to_memory(bus, memory)
```

Retention rules control how many entries are kept per tag when the maintenance
thread compacts the store:

```bash
export GPT_MEMORY_RETENTION="insight=40,error_fix=20"
```

### Persisting across runs

Reusing the same database path keeps interactions for future sessions:

```bash
# first run
python - <<'PY'
from gpt_memory import GPTMemoryManager
mgr = GPTMemoryManager("persistent.db")
mgr.log_interaction("hello", "world", tags=["insight"])
mgr.close()
PY

# later run
python - <<'PY'
from gpt_memory import GPTMemoryManager
mgr = GPTMemoryManager("persistent.db")
print([e.response for e in mgr.search_context("hello")])
PY
```

`get_similar_entries()` returns scored matches while `compact()` summarises and
prunes old rows according to a retention policy.  Use `close()` when finished to
ensure the connection is cleanly closed.

## GPTMemory wrapper

`GPTMemory` is a thin adapter around `MenaceMemoryManager`.  It is now
deprecated in favour of using any backend that implements
`GPTMemoryInterface` (typically `GPTMemoryManager`).  Existing code should
migrate to the interface and remove uses of this wrapper.

## Example integrations

- **Self-Coding Engine** – pass a `GPTMemoryManager` via the `gpt_memory`
  parameter.  Every LLM call is logged and previous interactions can be searched
  when proposing patches.
- **Learning & Self-Learning engines** – `GPTMemory` wraps the shared
  `MenaceMemoryManager`.  The self-learning service periodically prunes GPT
  logs while the learning engines retrain incrementally on `memory:new` events.
  Existing integrations should migrate to `GPTMemoryManager` or another
  `GPTMemoryInterface` implementation.

Use an in-memory database (`db_path=":memory:"`) for ephemeral runs or supply a
preloaded `SentenceTransformer` as `embedder` to enable semantic similarity
queries.

## Automatic maintenance

When `run_autonomous.py` launches it spawns a lightweight background thread that
periodically compacts the memory store using
`GPTMemoryManager.compact()`. Retention limits are configured with the
`GPT_MEMORY_RETENTION` environment variable which accepts comma separated
`tag=count` pairs, for example:

```bash
export GPT_MEMORY_RETENTION="error_fix=50,feedback=20"
```

The compaction interval defaults to one hour and can be customised via
`GPT_MEMORY_COMPACT_INTERVAL` (seconds). If no retention rules are provided the
maintenance thread is disabled.

## Automatic insight refresh

When using :func:`memory_logging.log_with_tags` with a
`LocalKnowledgeModule`, each successful log triggers a background refresh of
stored insights.  This keeps summarised feedback, fixes and improvement paths
up to date without blocking the caller.  Set the `GPT_AUTO_REFRESH_INSIGHTS`
environment variable to `0`/`false` to disable this behaviour if periodic
updates are not required.
