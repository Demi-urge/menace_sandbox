# Meta Workflow Planner

The `MetaWorkflowPlanner` encodes workflows into fixed-size vectors combining
structural information, recent ROI trends and semantic tokens. Embeddings are
persisted with `vector_utils.persist_embedding` so they can be clustered or
retrieved later.

## Usage

```python
import networkx as nx
from meta_workflow_planner import MetaWorkflowPlanner

# Build a minimal dependency graph
g = nx.DiGraph()
g.add_edge("A", "B")
planner = MetaWorkflowPlanner(graph=type("G", (), {"graph": g})())

workflow = {"workflow": ["fetch", "process", "store"]}
vector = planner.encode("A", workflow)
```

The embedding is written to `embeddings.jsonl` in the current working directory
along with metadata such as the ROI curve and dependency depth.

## Clustering and chaining

Embeddings can be used to group workflows or to assemble simple pipelines:

```python
workflows = {
    "A": {"workflow": ["fetch"]},
    "B": {"workflow": ["process"]},
    "C": {"workflow": ["store"]},
}

# Cluster similar workflow specs
clusters = planner.cluster_workflows(workflows, threshold=0.8)

# Build a high‑synergy chain starting from A
pipeline = planner.compose_pipeline("A", workflows, length=3)
```

`cluster_workflows` encodes each specification and groups workflows using
ROI‑weighted cosine similarity, while `compose_pipeline` iteratively selects the
best next step based on `WorkflowSynergyComparator` scores scaled by recent ROI
trends.  The scoring formula is ``synergy_score * (1 + ROI)`` and the
``synergy_weight`` and ``roi_weight`` parameters expose knobs to tune the
influence of each component.

## Sandbox simulation

`plan_and_validate` executes suggested chains inside a sandbox to measure ROI
gains and robustness:

```python
def step_a():
    return 1.0

def step_b():
    return 2.0

records = planner.plan_and_validate([0.0], {"a": step_a, "b": step_b}, top_k=1)
```

Each record contains the chain, total ROI gain, failure count and entropy. The
method accepts a custom runner or will instantiate a default
`WorkflowSandboxRunner` when available.

## Configuration

`MetaWorkflowPlanner` accepts several knobs to tailor the generated vectors:

- `max_functions`, `max_modules`, `max_tags` – size of the one-hot sections for
  function names, module identifiers and tags.
- `roi_window` – number of recent ROI measurements incorporated into the
  embedding.
- `function_index`, `module_index`, `tag_index` – optional dictionaries used to
  maintain stable token indices across planner instances.
- `graph` and `roi_db` – optional helpers providing structural context and ROI
  history. When omitted, lightweight defaults are created.
- `cluster_workflows(threshold)` controls the similarity cutoff for the
  ROI‑weighted similarity matrix when grouping workflow identifiers.
- `compose_pipeline(length, synergy_weight, roi_weight)` limits the number of
  steps in generated chains and exposes weights to balance synergy against ROI.
- `plan_and_validate(top_k, failure_threshold, entropy_threshold)` governs the
  number of candidate chains considered and the acceptance criteria during
  sandbox execution.

The resulting embedding structure is:

1. Dependency depth and branching factor.
2. ROI curve of length `roi_window`.
3. One-hot vectors for functions, modules and tags.
