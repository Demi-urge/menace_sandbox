# Workflow Graph

The workflow graph models the project's workflows as a directed acyclic graph. Each node represents a workflow and edges capture dependency relationships that let the system analyse how changes ripple through the network and persist this information on disk.

## Building and Persisting the DAG

`WorkflowGraph` instances can be created from scratch or pre-populated from
`task_handoff_bot.WorkflowDB`.  Workflows are added with `add_workflow` and
dependencies with `add_dependency`.  Every mutation triggers a save to
`sandbox_data/workflow_graph.json` (or a custom path if provided) so the graph
can be reconstructed on the next run.

```python
from workflow_graph import WorkflowGraph

g = WorkflowGraph()
g.add_workflow("A", roi=0.5)
g.add_dependency("A", "B", impact_weight=0.6)
g.save()  # optional, mutations save automatically
```

## Edge Weights

Dependency edges carry an aggregate `impact_weight` along with
`resource_weight`, `data_weight` and `logical_weight` attributes reflecting how
strongly one workflow influences another through different channels. The
weights are derived by [`estimate_edge_weight`](../workflow_graph.py) which
blends structural heuristics with live telemetry:

1. **Resource overlap** – shared bots or queues.
2. **API/module similarity** – common steps or `action_chains`, optionally enhanced with vector similarities.
3. **Output coupling** – the output of one workflow feeding into another.
4. **ROI correlation** – historical return‑on‑investment deltas from `roi_tracker`.
5. **Queue load overlap** – allocation history from `resource_allocation_bot`.

Each signal contributes a normalised value in `[0, 1]` and the average becomes
the final `impact_weight`. Runtime metrics are included on a best‑effort basis;
if ROI logs or allocation data are missing the calculation transparently falls
back to the structural heuristics.

## Running `simulate_impact_wave`

```python
from workflow_graph import WorkflowGraph

g = WorkflowGraph()
g.add_workflow("A", roi=0.5)
g.add_workflow("B", roi=0.3)
g.add_dependency("A", "B", resource_weight=0.8, data_weight=0.2)
projection = g.simulate_impact_wave("A", 0.1, 0.0, resource_damping=0.5)
print(projection["B"])  # {'roi': 0.04, 'synergy': 0.0}
```

Running the above propagates projected ROI and synergy *deltas* from workflow `A`
to its dependants using the stored weights.  The optional damping parameters
allow tuning how strongly each dependency type influences the final result. The result maps each
affected workflow to the simulated change in metrics which downstream
self‑improvement modules can consume.

## Dependencies and Fallback

[`workflow_graph.py`](../workflow_graph.py) uses [`NetworkX`](https://networkx.org) when available for graph management. If the library is missing the module transparently falls back to a lightweight adjacency-list implementation so basic functionality remains available.

`simulate_impact_wave` accepts explicit ROI and synergy deltas and does not rely on external predictors or history databases.

## Publishing Workflow Events

Other components should emit workflow lifecycle events on a shared
`UnifiedEventBus` so the graph stays current.  After calling
`graph.attach_event_bus(bus)`, the graph listens for the following topics:

* `workflows:new` – payload contains `workflow_id` of a new workflow.
* `workflows:update` – payload may include `workflow_id`, `roi`,
  `synergy_scores`, and optional `roi_delta` or `synergy_delta` values.
* `workflows:deleted` / `workflows:refactor` – remove a workflow from the DAG.

```python
from unified_event_bus import UnifiedEventBus

bus = UnifiedEventBus()
graph.attach_event_bus(bus)
bus.publish("workflows:new", {"workflow_id": "42"})
```

Publishing these events ensures the persisted DAG reflects the live system.
