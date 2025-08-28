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

The resulting embedding structure is:

1. Dependency depth and branching factor.
2. ROI curve of length `roi_window`.
3. One-hot vectors for functions, modules and tags.
