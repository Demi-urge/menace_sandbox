# Cross Query Utilities

`cross_query` combines information from the different Menace databases. The
module exposes helpers to discover relationships between bots, workflows and code
templates using `BotRegistry`, `PathwayDB` and `MenaceDB`.

## Related Workflows

```python
from menace.cross_query import related_workflows

workflows = related_workflows(
    "ExampleBot",
    registry=registry,
    menace_db=menace_db,
    pathway_db=pathway_db,
)
```

The function returns workflow names linked to the given bot. Connections are
expanded via `BotRegistry` edges and ranked with myelination scores from
`PathwayDB`.

## Similar Code Snippets

```python
from menace.cross_query import similar_code_snippets

snippets = similar_code_snippets(
    "data processor",
    menace_db=menace_db,
    registry=registry,
    pathway_db=pathway_db,
)
```

This helper searches the `code` table for matching summaries or template types
and also includes snippets used by connected bots. When a `PathwayDB` instance is
supplied, similarity scores are derived from previously recorded pathways.

## Related Resources

```python
from menace.cross_query import related_resources

resources = related_resources(
    "ExampleBot",
    registry=registry,
    menace_db=menace_db,
    info_db=info_db,
    memory_mgr=memory_mgr,
    pathway_db=pathway_db,
)
```

`related_resources` expands on the other helpers by traversing bot connections,
workflow links, stored research items and memory entries. Memory is queried by
tags as well as by any stored `bot_id` or `info_id`. The returned dictionary
contains associated bot names, workflow titles, research summaries and memory
keys ranked using myelination scores when `PathwayDB` is provided.

## Workflow Features for New Entries

```python
from menace.cross_query import entry_workflow_features

features = entry_workflow_features(
    {"bot": "ExampleBot"},
    registry=registry,
    menace_db=menace_db,
    pathway_db=pathway_db,
)
```

`entry_workflow_features` links a freshly inserted memory item, code snippet or
research record to existing workflows. Bot names are extracted from the entry
and expanded via `BotRegistry` before calling `related_workflows`. When the
entry provides a `summary` field the helper also uses `similar_code_snippets`
to connect related bots. The returned list can feed directly into learning
models so that new database items immediately influence planning decisions.

## Workflow ROI Statistics

```python
from menace.cross_query import workflow_roi_stats, rank_workflows

stats = workflow_roi_stats(
    "ExampleWorkflow",
    roi_db=roi_db,
    metrics_db=metrics_db,
)

ranking = rank_workflows([
    "ExampleWorkflow",
    "AnotherWorkflow",
], roi_db=roi_db, metrics_db=metrics_db)
```

`workflow_roi_stats` aggregates ROI history stored by `ResourceAllocationOptimizer`
and evaluation metrics recorded in `MetricsDB`. The returned dictionary contains
total ROI, consumed CPU seconds and API cost for a workflow. `rank_workflows`
applies this helper across multiple workflow names and returns them ordered by
ROI per CPU second.

## Full-Text Search in InfoDB

`InfoDB` now attempts to create an optional FTS5 table named `info_fts` when the
database is initialised. The virtual table indexes the `title`, `tags` and
`content` columns from `info` so that searches can use SQLite's full-text
capabilities. `InfoDB.search()` automatically queries `info_fts` when it exists
and falls back to the previous `LIKE` query when FTS5 is unavailable.

## Registry Persistence

`BotRegistry` can be initialised with a `persist` path and `UnifiedEventBus`
instance. When enabled, every new bot or interaction automatically triggers a
save to the SQLite file and publishes update events so other components can keep
their graphs in sync across restarts. Heartbeat tracking allows `BotRegistry`
to report currently active bots and last-seen times, while interaction metadata
such as duration and success rates is aggregated for performance analysis.
