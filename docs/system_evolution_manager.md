# System Evolution Manager

`SystemEvolutionManager` coordinates global genetic algorithm (GA) evolution across multiple bots and uses `StructuralEvolutionBot` to suggest structural improvements.

## Initialization

```python
from menace.system_evolution_manager import SystemEvolutionManager

manager = SystemEvolutionManager(["bot_a", "bot_b"], metrics_db=my_metrics)
```

Provide a list of bot names and optionally a `MetricsDB` instance where ROI metrics will be logged.

## `run_if_signals`

```python
maybe_result = manager.run_if_signals(error_rate=0.25, energy=0.2,
                                      error_thresh=0.2, energy_thresh=0.3)
```

This method checks the current error rate and energy score against configured thresholds and
compares them to the previous cycle. If degradation is detected it calls `run_cycle` and
returns its `EvolutionCycleResult`; otherwise it returns `None`.

## `run_cycle`

```python
result = manager.run_cycle()
print(result.ga_results)
for prediction in result.predictions:
    print(prediction)
```

`run_cycle` takes a snapshot of the system, obtains change predictions from
`StructuralEvolutionBot`, and runs a GA learning cycle for each registered bot
via `GALearningManager`. The average ROI is stored in `MetricsDB` under
`"evolution_cycle"`. The returned `EvolutionCycleResult` contains both the GA
ROI values and the structural predictions.

## Major change approval

`StructuralEvolutionBot.apply_major_change` no longer prompts for user input.
Instead it accepts an optional callback or uses `EvolutionApprovalPolicy` to
automatically approve changes. The default policy approves when the predicted
impact is below a configurable threshold.
