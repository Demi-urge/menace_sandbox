# Workflow Synergy Comparator

`WorkflowSynergyComparator` analyses two workflow specifications and derives structural metrics to guide merge and duplicate decisions.

## Comparison metrics

- **Similarity** – cosine similarity between workflow embeddings. High similarity implies nearly identical structure.
- **Shared modules** – count of overlapping modules between the two workflow graphs.
- **Entropy** – Shannon entropy of each workflow's module distribution, highlighting structural complexity.

## Configuration

`WorkflowSynergyComparator.is_duplicate(result, similarity_threshold, entropy_threshold)` evaluates a
pre-computed :class:`ComparisonResult`:

- `similarity_threshold` – minimum cosine similarity to treat workflows as near duplicates (default `0.95`).
- `entropy_threshold` – maximum allowed absolute entropy delta (default `0.05`).

`WorkflowSynergyComparator.merge_duplicate(base_id, dup_id, out_dir)` merges a duplicate workflow into a
canonical base and refreshes lineage summaries.

## Integration

`WorkflowEvolutionManager` invokes `WorkflowSynergyComparator.compare()` when scoring variant workflows. If the similarity exceeds `settings.workflow_merge_similarity` and the entropy delta stays below `settings.workflow_merge_entropy_delta`, the manager merges the variant using the workflow merger.

