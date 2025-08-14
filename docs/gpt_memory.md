# GPT Memory

`gpt_memory.py` provides lightweight persistence for GPT interactions.  It can
store prompts and responses with tags, optionally embed each prompt for semantic
search and compact old records into concise summaries.

## GPTMemoryManager

```python
from gpt_memory import GPTMemoryManager

mgr = GPTMemoryManager("memory.db")
mgr.log_interaction("user question", "assistant reply", tags=["note"])
entries = mgr.search_context("question")
```

Key configuration options:

- `db_path` – SQLite file used for storage (`"gpt_memory.db"` by default).
- `embedder` – optional `SentenceTransformer` model enabling semantic search.

`get_similar_entries()` returns scored matches while `compact()` summarises and
prunes old rows according to a retention policy.  Use `close()` when finished to
ensure the connection is cleanly closed.

## GPTMemory wrapper

`GPTMemory` is a thin adapter around `MenaceMemoryManager`.  It exposes
`log_interaction()`, `fetch_context()`, `summarize_and_prune()` and `retrieve()`
so existing components can store GPT conversations without depending on the
SQLite backend.

## Example integrations

- **Self-Coding Engine** – pass a `GPTMemoryManager` via the `gpt_memory`
  parameter.  Every LLM call is logged and previous interactions can be searched
  when proposing patches.
- **Learning & Self-Learning engines** – `GPTMemory` wraps the shared
  `MenaceMemoryManager`.  The self-learning service periodically prunes GPT
  logs while the learning engines retrain incrementally on `memory:new` events.

Use an in-memory database (`db_path=":memory:"`) for ephemeral runs or supply a
preloaded `SentenceTransformer` as `embedder` to enable semantic similarity
queries.
