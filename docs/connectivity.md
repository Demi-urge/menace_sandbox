# Menace Interconnectivity Guide

This document explains how the Menace bots, databases and event system should
link together for maximum emergent behaviour.

## Database Router

Bots should interact with data through `DBRouter`. The router decides whether
tables reside in the local or shared SQLite database and returns a
``LoggedConnection`` via :meth:`get_connection`. These wrappers automatically
log the number of rows read or written for each statement. The legacy
`database_router` module has been removed; import `DBRouter` from `db_router`
instead:

```python
from menace.db_router import DBRouter

router = DBRouter("demo", "./local.db", "./shared.db")
conn = router.get_connection("bots")
```

## Event Bus

`UnifiedEventBus` can persist events to a SQLite log when given a path. Bots can
replay events using `replay()` if they were offline. The bus now supports
optional asyncio callbacks so bots may react to events in real time when an
event loop is provided. A new `NetworkedEventBus` implementation leverages
RabbitMQ so distributed bots share a common event stream. Listeners such as
`Neo4jGraphListener` can subscribe to bus topics and update external systems in
real time.
Database changes are emitted on `cdc:<table>` topics so bots can react
immediately to inserts, updates and deletes.

## Memory Manager

`MenaceMemoryManager` stores versioned entries and can associate them with bots
or research items. Use `store(..., bot_id=<id>, info_id=<id>)` when adding
contextual memory so that cross queries can locate relevant history. The memory
database includes an FTS5 table enabling fast full text search via
`MenaceMemoryManager.search()`. When the optional SentenceTransformer
dependency is available, embeddings are stored for each entry so that
`query_vector()` can perform semantic search using cosine similarity.

## EmbeddableDBMixin

Several SQLite databases now inherit from `EmbeddableDBMixin` to persist
vector embeddings alongside their primary tables.  The mixin stores vectors
in a shared `embeddings` table and supports FAISS or Annoy for similarity
search.

| Database | Embedded fields |
|----------|-----------------|
| `BotDB` | `purpose`, `tags`, `toolchain` |
| `WorkflowDB` | assigned bots, enhancements, title, description, tags, category, type, status |
| `ErrorDB` | `message` |
| `EnhancementDB` | `before_code`, `after_code`, `summary` |
| `InfoDB` | title/topic, summary, content, tags, category, type, associated bots, associated errors, performance data, source URL, notes |

Select the backend with the `vector_backend` argument (`"faiss"` or
`"annoy"`) and index location via `vector_index_path`.  Setting
`VECTOR_BACKEND` and `VECTOR_INDEX_PATH` environment variables provides
defaults.  Existing databases can populate vectors with
`backfill_embeddings()`.

For ad-hoc SQL needs obtain a connection and execute statements directly:

```python
conn = router.get_connection("bots")
rows = conn.execute("SELECT name FROM bots WHERE status=?", ("active",)).fetchall()
```

## Bot Heartbeat Tracking

`BotRegistry` exposes a lightweight heartbeat mechanism so that other
components can track which bots are currently running. Each bot should call
`record_heartbeat()` regularly—usually once per main loop iteration. When an
event bus is attached, the call publishes a `bot:heartbeat` event that
monitoring tools can subscribe to:

```python
registry = BotRegistry(event_bus=bus)
registry.record_heartbeat("ExampleBot")
bus.subscribe("bot:heartbeat", handle_heartbeat)
```

Publication failures are logged and emitted as `bot:heartbeat_error` events.

Active bots can be queried via `active_bots(timeout)` which returns a mapping of
bot names to their last-seen timestamp. Only bots seen within the supplied
timeout are returned:

```python
for name, ts in registry.active_bots(timeout=300).items():
    print(name, ts)
```

## Bot Update Guardrails

`BotRegistry.hot_swap_bot` verifies that every update is accompanied by
provenance metadata. If a bot is updated without a `patch_id` or commit hash, or
if the metadata does not match the recorded patch, the reload is aborted and a
`bot:update_blocked` event is emitted. Administrators must perform a manual
override and retry the operation once the provenance is confirmed.
