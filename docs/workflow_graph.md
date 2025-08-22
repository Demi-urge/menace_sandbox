# Workflow Graph

The workflow graph models the project's workflows as a directed acyclic graph. Each node represents a workflow and edges capture dependency relationships that let the system analyse how changes ripple through the network and persist this information on disk.

## Edge Weights

Dependency edges carry an `impact_weight` reflecting how strongly one workflow influences another. The weight is derived by [`estimate_edge_weight`](../workflow_graph.py) which blends three heuristics:

1. **Resource overlap** – shared bots or queues.
2. **API/module similarity** – common steps or `action_chains`, optionally enhanced with vector similarities.
3. **Output coupling** – the output of one workflow feeding into another.

The final weight is normalised to the range `[0, 1]` and falls back to `1.0` when required supporting data or modules are unavailable.

## Example: `simulate_impact_wave`

```python
from workflow_graph import WorkflowGraph

g = WorkflowGraph()
g.add_workflow("A", roi=0.5)
g.add_workflow("B", roi=0.3)
g.add_dependency("A", "B", impact_weight=0.6)
projection = g.simulate_impact_wave("A", 0.1, 0.0)
print(projection["B"])  # {'roi': 0.06, 'synergy': 0.0}
```

Running the above propagates projected ROI and synergy *deltas* from workflow `A`
to its dependants using the stored `impact_weight` values.  The result maps each
affected workflow to the simulated change in metrics which downstream
self‑improvement modules can consume.

## Dependencies and Fallback

[`workflow_graph.py`](../workflow_graph.py) uses [`NetworkX`](https://networkx.org) when available for graph management. If the library is missing the module transparently falls back to a lightweight adjacency-list implementation so basic functionality remains available.

`simulate_impact_wave` accepts explicit ROI and synergy deltas and does not rely on external predictors or history databases.
