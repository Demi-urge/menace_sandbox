# Menace Interconnectivity Guide

This document explains how the Menace bots, databases and event system should
link together for maximum emergent behaviour.

## Database Router

Bots should interact with data through `DatabaseRouter`. The router mirrors
inserts and updates across the local SQLite stores and the optional
`MenaceDB`. It also subscribes to events so that new research items
automatically create cross references to existing bots. When a `remote_url`
is supplied the router opens a pooled connection to the central database and
replicates writes atomically using `TransactionManager`.

When no connection to the remote database is available the router works
offline using the local stores. Results of `execute_query` and the cross
query helpers are cached in-memory for `cache_seconds` (60&nbsp;seconds by
default). Use `router.flush_cache()` or `router.enable_cache(False)` to
disable caching when you need fresh reads.

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

## Cross Query Helpers

Functions in `menace.cross_query` explore these links to surface related
workflows, code snippets and resources across the system. The
`related_resources` helper traverses bots, research items and memory entries to
expose higher-order connections.

These helpers are also available directly from `DatabaseRouter`:

```python
router = DatabaseRouter(menace_db=menace_db, info_db=info_db, memory_mgr=memory_mgr)
workflows = router.related_workflows("ExampleBot", registry=registry, pathway_db=pathway_db)
snippets = router.similar_code_snippets("data processor", registry=registry)
resources = router.related_resources("ExampleBot", registry=registry, pathway_db=pathway_db)
```

For ad-hoc SQL needs a centralised API is available:

```python
rows = router.execute_query("bot", "SELECT name FROM bots WHERE status=?", ["active"])
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
