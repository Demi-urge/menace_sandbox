# Workflow Synergy Comparator

`WorkflowSynergyComparator` analyses two workflow specifications and derives
rich structural metrics to guide merge and duplicate decisions.

## Comparator logic

The comparator loads each workflow, extracts its module sequence and embeds a
lightweight graph.  It computes several metrics:

- **Similarity** – cosine similarity between workflow embeddings.  High
  similarity implies nearly identical structure.
- **Shared modules** – Jaccard ratio of overlapping modules.
- **Entropy / expandability** – Shannon entropy of each workflow's module
  distribution with the mean of both entropies serving as an expandability
  score.
- **Efficiency** – ROI based efficiency for the combined workflow graph.
 - **Modularity** – community modularity of the combined workflow graph, or the
   ratio of unique modules to total steps when community detection is
   unavailable.

The weighted mean of these metrics becomes an aggregate score.  Workflows
qualify for merging when their similarity meets
`SandboxSettings.workflow_merge_similarity` and the entropy gap stays below
`SandboxSettings.workflow_merge_entropy_delta`.

Additionally, `compare` attaches overfitting reports (`overfit_a` / `overfit_b`)
highlighting low entropy or excessive module repetition.  Call
`WorkflowSynergyComparator.analyze_overfitting` directly to inspect a single
workflow and to update the best‑practices repository when no overfitting is
detected.

## Usage

```python
from menace.workflow_synergy_comparator import WorkflowSynergyComparator

scores = WorkflowSynergyComparator.compare("123", "456")
print(scores.similarity, scores.efficiency)

if scores.overfit_a.is_overfitting():
    print("first workflow shows overfitting signals")

is_dup = WorkflowSynergyComparator.is_duplicate(
    scores, {"similarity": 0.95, "entropy": 0.05}
)
``is_duplicate`` expects the ``SynergyScores`` returned by ``compare``.

Command‑line usage is available through `workflow_synergy_cli.py`:

```bash
python workflow_synergy_cli.py 123 456 --out result.json
```

## Configuration

`WorkflowSynergyComparator.merge_duplicate(base_id, dup_id, out_dir)` merges a
duplicate workflow into a canonical base and refreshes lineage summaries.
`merge_duplicate` is also available as a module level function.

Tune merge and duplicate thresholds via environment variables:

```bash
export WORKFLOW_MERGE_SIMILARITY=0.9
export WORKFLOW_MERGE_ENTROPY_DELTA=0.1
export WORKFLOW_DUPLICATE_SIMILARITY=0.95
export WORKFLOW_DUPLICATE_ENTROPY=0.05
```

For overfitting checks the `analyze_overfitting` helper accepts
`entropy_threshold` and `repeat_threshold` parameters (defaults: `1.0` and `3`).

## Integration

`WorkflowEvolutionManager` invokes `WorkflowSynergyComparator.compare()` when
scoring variant workflows.  If the similarity exceeds
`settings.workflow_merge_similarity` and the entropy delta stays below
`settings.workflow_merge_entropy_delta`, the manager merges the variant using
the workflow merger.

