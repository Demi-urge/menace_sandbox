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

## Diminishing-returns gating

Workflow ROI improvements are tracked with an exponential moving average (EMA).
Each workflow's EMA and consecutive failure count are persisted to
`sandbox_data/workflow_roi_ema.json`, allowing the gate to survive process
restarts. The EMA is updated with `0.3 * delta + 0.7 * previous_ema` on each
cycle. When the EMA remains below `ROI_GATING_THRESHOLD` for
`ROI_GATING_CONSECUTIVE` consecutive runs, `is_stable()` returns `True` and
further evolution is skipped.

The counter resets automatically whenever the EMA rises above the threshold or a
variant is promoted. Remove `sandbox_data/workflow_roi_ema.json` or clear the
workflow from `WorkflowStabilityDB` to reset the gate manually.

Tune the gating behaviour with environment variables:

```bash
export ROI_GATING_THRESHOLD=0.05
export ROI_GATING_CONSECUTIVE=5
```

## Enabling in self-improvement cycles

`SelfImprovementEngine` embeds a `WorkflowEvolutionManager` by default. Populate `PathwayDB` with recent workflow sequences, register the engine and run all cycles:

```python
from menace.self_improvement_engine import SelfImprovementEngine, ImprovementEngineRegistry

engine = SelfImprovementEngine(bot_name="alpha")
registry = ImprovementEngineRegistry()
registry.register_engine("alpha", engine)
registry.run_all_cycles()
```

This executes baseline runs, benchmarks generated variants and promotes the best-performing workflow.
