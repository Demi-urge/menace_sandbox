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
from vector_service import Retriever
from vector_service.context_builder import ContextBuilder
workflows = {
    "A": {"workflow": ["fetch"]},
    "B": {"workflow": ["process"]},
    "C": {"workflow": ["store"]},
}

# Cluster similar workflow specs
builder = ContextBuilder()
retr = Retriever(context_builder=builder)
clusters = planner.cluster_workflows(
    workflows, retriever=retr, epsilon=0.8, context_builder=planner.context_builder
)

# Build a high‑synergy chain starting from A
pipeline = planner.compose_pipeline(
    "A", workflows, length=3, context_builder=planner.context_builder
)
```

`cluster_workflows` encodes each specification, persists the embedding and
groups workflows using ROI‑weighted cosine similarity.  When
[scikit‑learn](https://scikit-learn.org) is available the normalised distance
matrix is clustered with DBSCAN via the provided `Retriever`.  Without
scikit‑learn a simple similarity‑threshold fallback groups connected workflows
so the planner remains functional in lightweight environments.  In both cases
`compose_pipeline` iteratively blends embedding similarity,
`WorkflowSynergyComparator` scores and recent ROI trends to choose the next
step.  The default formula is
``(similarity * similarity_weight + synergy * synergy_weight) * (1 + ROI * roi_weight)``
giving callers independent control over similarity, synergy and ROI.  Setting
`synergy_weight` to `0` ignores structural synergy and ranks steps purely by
similarity and ROI.

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

## Persistent reinforcement

`MetaWorkflowPlanner` persists reinforcement signals after each chain execution
by recording the chain, ROI gain, failure count and entropy in
`synergy_history.db`.  On startup the planner reloads this history to seed its
`cluster_map`, allowing previously successful (or failing) chains to influence
future planning sessions.

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
- `cluster_workflows(retriever, epsilon, min_samples)` groups workflow
  identifiers using ROI‑weighted similarity and DBSCAN. ``epsilon`` and
  ``min_samples`` control the clustering density.
- `compose_pipeline(length, similarity_weight, synergy_weight, roi_weight)` limits
  the number of steps in generated chains and exposes weights to balance
  embedding similarity, structural synergy and ROI.
- `plan_and_validate(top_k, failure_threshold, entropy_threshold)` governs the
  number of candidate chains considered and the acceptance criteria during
  sandbox execution.

The resulting embedding structure is:

1. Dependency depth and branching factor from the workflow graph.
2. ROI curve of length `roi_window` representing recent gains.
3. Code context depth, branching and average code ROI curve.
4. TF‑IDF weighted function tokens, followed by per‑function average ROI and
   failure ratios.
5. One‑hot module tokens.
6. TF‑IDF weighted tag tokens.
