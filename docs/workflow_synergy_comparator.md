# Workflow Synergy Comparator

`WorkflowSynergyComparator` analyses two workflow specifications and derives
structural metrics to guide merge and duplicate decisions.

## Comparator logic

The comparator loads each workflow, extracts its module sequence and embeds a
lightweight graph.  It then computes three core metrics:

- **Similarity** – cosine similarity between workflow embeddings.  High
  similarity implies nearly identical structure.
- **Shared modules** – Jaccard ratio of overlapping modules.
- **Entropy** – Shannon entropy of each workflow's module distribution,
  highlighting structural complexity.

The mean of similarity, shared module ratio and average entropy (called
**expandability**) becomes an aggregate score.  Workflows qualify for merging
when their similarity meets `SandboxSettings.workflow_merge_similarity` and the
entropy gap stays below `SandboxSettings.workflow_merge_entropy_delta`.

## Usage

```python
from menace.workflow_synergy_comparator import WorkflowSynergyComparator

scores = WorkflowSynergyComparator.compare("123", "456")
print(scores.similarity, scores.expandability)

is_dup = WorkflowSynergyComparator.is_duplicate(
    "123", "456", {"similarity": 0.95, "entropy": 0.05}
)
```

## Configuration

`WorkflowSynergyComparator.merge_duplicate(base_id, dup_id, out_dir)` merges a
duplicate workflow into a canonical base and refreshes lineage summaries.

Tune merge and duplicate thresholds via environment variables:

```bash
export WORKFLOW_MERGE_SIMILARITY=0.9
export WORKFLOW_MERGE_ENTROPY_DELTA=0.1
export WORKFLOW_DUPLICATE_SIMILARITY=0.95
export WORKFLOW_DUPLICATE_ENTROPY=0.05
```

## Integration

`WorkflowEvolutionManager` invokes `WorkflowSynergyComparator.compare()` when
scoring variant workflows.  If the similarity exceeds
`settings.workflow_merge_similarity` and the entropy delta stays below
`settings.workflow_merge_entropy_delta`, the manager merges the variant using
the workflow merger.

