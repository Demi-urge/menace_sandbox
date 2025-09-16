# Workflow Evolution

`WorkflowEvolutionManager` benchmarks alternative workflow sequences and promotes improvements.

## Baseline execution

`evolve()` first executes the current workflow with `CompositeWorkflowScorer` to record runtime, success rate and ROI. These metrics are stored in `ROIResultsDB` for comparison.

## Variant generation

`WorkflowEvolutionBot.generate_variants()` proposes candidates through:

- **Step swaps** using synergistic modules.
- **Reordering** existing steps (including simple reversal).
- **Module injection** by appending intent-related modules.

Each candidate sequence is validated for structural soundness before evaluation.

## Benchmarking and promotion

Every variant is converted into a callable and scored by `CompositeWorkflowScorer`. The variant with the highest ROI gain is promoted when its ROI exceeds the baseline. Outcomes are logged via `ROIResultsDB` and `mutation_logger` for later analysis.

## Synergy comparison and merging

`WorkflowSynergyComparator` analyses the baseline and each variant for
structural similarity. It reports **efficiency** (embedding cosine
similarity), **modularity** (Jaccard overlap) and **expandability** (average
step entropy). When the mean of efficiency and modularity meets
`SandboxSettings.workflow_merge_similarity` and the entropy difference stays
below `SandboxSettings.workflow_merge_entropy_delta`, the manager attempts to
merge both specifications via `workflow_merger.merge_workflows`. The merged
workflow is re-scored and promoted if it outperforms the baseline.

Tune the merge thresholds with environment variables:

```bash
export WORKFLOW_MERGE_SIMILARITY=0.95
export WORKFLOW_MERGE_ENTROPY_DELTA=0.05
```

## Duplicate collapsing

After checking merge candidates, the evolution loop scans stable workflows for
near duplicates.  `WorkflowSynergyComparator.is_duplicate()` compares the
variant against each stable workflow and collapses it into the closest match
when:

- cosine similarity >= `SandboxSettings.duplicate_similarity`
- entropy gap <= `SandboxSettings.duplicate_entropy`

Enable or adjust the behaviour via environment variables:

```bash
export WORKFLOW_DUPLICATE_SIMILARITY=0.95
export WORKFLOW_DUPLICATE_ENTROPY=0.05
```

Example invocation pulling thresholds from the environment:

```python
import os
from menace.workflow_evolution_manager import evolve

os.environ["WORKFLOW_DUPLICATE_SIMILARITY"] = "0.95"
os.environ["WORKFLOW_DUPLICATE_ENTROPY"] = "0.05"

promoted = evolve(workflow_callable, workflow_id)
```

## Diminishing-returns gating

Workflow ROI improvements are tracked with an exponential moving average (EMA).
Each workflow's EMA and consecutive failure count are stored alongside stability
data in `WorkflowStabilityDB`, allowing the gate to survive process restarts.
The EMA is updated with `alpha * delta + (1 - alpha) * previous_ema` where
`alpha` defaults to ``SandboxSettings.roi_ema_alpha`` (configurable via
`ROI_EMA_ALPHA`). When the EMA remains below `ROI_GATING_THRESHOLD` for
`ROI_GATING_CONSECUTIVE` consecutive runs, `is_stable()` returns `True` and
further evolution is skipped.

The counter resets automatically whenever the EMA rises above the threshold or a
variant is promoted. Clear the workflow from `WorkflowStabilityDB` to reset the
gate manually.

Tune the gating behaviour with environment variables:

```bash
export ROI_GATING_THRESHOLD=0.05
export ROI_GATING_CONSECUTIVE=5
# Optional smoothing factor for the EMA
export ROI_EMA_ALPHA=0.1
```

## Enabling in self-improvement cycles

`SelfImprovementEngine` embeds a `WorkflowEvolutionManager` by default. Populate `PathwayDB` with recent workflow sequences, register the engine and run all cycles:

```python
from menace.self_improvement.api import SelfImprovementEngine, ImprovementEngineRegistry
from context_builder_util import create_context_builder

engine = SelfImprovementEngine(
    context_builder=create_context_builder(),
    bot_name="alpha",
)
registry = ImprovementEngineRegistry()
registry.register_engine("alpha", engine)
registry.run_all_cycles()
```

`run_all_cycles()` delegates to `WorkflowEvolutionManager.evolve()` after the
refactor, so each cycle benchmarks variants, promotes improvements and marks
stable workflows automatically.
