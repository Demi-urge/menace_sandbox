# Knowledge Graph

`KnowledgeGraph` links bots, code paths and insights using a lightweight
`networkx` directed graph.  The graph can ingest GPT memories and other
telemetry so services share context across sessions.

## Listening for memory events

`GPTMemoryManager.log_interaction` publishes a `"memory:new"` event whenever a
prompt/response pair is stored.  `KnowledgeGraph.listen_to_memory` subscribes to
those events and calls `ingest_gpt_memory` to map the new insight into the
graph:

```python
from unified_event_bus import UnifiedEventBus
from gpt_memory import GPTMemoryManager
from knowledge_graph import KnowledgeGraph

bus = UnifiedEventBus()
mem = GPTMemoryManager("persistent.db", event_bus=bus)
graph = KnowledgeGraph("kg.gpickle")

graph.listen_to_memory(bus, mem)  # updates graph on each memory:new event
```

Both the SQLite database and `kg.gpickle` file are reused across runs, enabling
persistent cross-session learning.  Compaction settings are inherited from
`GPTMemoryManager`; for example:

```bash
export GPT_MEMORY_RETENTION="insight=40,error_fix=20"
```

After the environment variable is set the maintenance thread ensures only the
most recent memories for each tag are retained while the knowledge graph stays
up to date through the event bus.

## Sandbox warning about persisted graphs

The autonomous sandbox bundles a lightweight stub of `networkx` so that unit
tests do not need the full dependency tree.  The stub intentionally raises when
code accesses advanced module attributes such as `__path__`, which prevents the
graph loader from deserializing a persisted `knowledge_graph.gpickle` file.  As a
result `KnowledgeGraph` prints a warning about skipping the saved graph on each
startup.  This is expected in the sandbox environment; install the real
`networkx` package locally if you want to load the persisted graph instead of
seeing the warning.
