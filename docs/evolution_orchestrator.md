# Evolution Orchestrator

The `EvolutionOrchestrator` coordinates different improvement routines based on
metrics from `DataBot` and financial signals from `CapitalManagementBot`.
When error rates exceed a configured threshold or the system energy score drops
below a limit, the orchestrator triggers either the `SelfImprovementEngine` or
the [`SystemEvolutionManager`](system_evolution_manager.md).
If ROI improves beyond a configured delta, new bots can be spawned via
`BotCreationBot`. Each run updates `ResourceAllocationOptimizer` to keep
resources focused on the best performing bots.

Each cycle is stored in `EvolutionHistoryDB`, recording ROI before and after the
change so that future strategies can learn from past outcomes.

When an optional `EvolutionAnalysisBot` or `EvolutionPredictor` is supplied, the
orchestrator queries it for the expected ROI of candidate actions such as
`self_improvement` and `system_evolution`. If multiple triggers fire or metrics
hover around the configured thresholds, the action with the highest predicted
ROI is chosen.

```python
from menace.evolution_orchestrator import EvolutionOrchestrator, EvolutionTrigger

orchestrator = EvolutionOrchestrator(
    data_bot=my_data_bot,
    capital_bot=my_capital_bot,
    improvement_engine=improvement,
    evolution_manager=sys_evo,
    triggers=EvolutionTrigger(error_rate=0.2, roi_drop=-0.5, energy_threshold=0.4),
    bot_creator=creator,
    resource_optimizer=optimizer,
)

# Check conditions and evolve if necessary
orchestrator.run_cycle()
```

## Automatic scheduling

`EvolutionScheduler` can monitor metrics continuously and invoke
`orchestrator.run_cycle()` whenever error rates or the energy score cross the
configured thresholds. Detected anomalies from `OperationalMonitoringBot` are
checked against a configurable count threshold and the scheduler keeps a moving
average of `DataBot.engagement_delta()`. If the anomaly count or engagement
trend drops below the configured limits, the orchestrator is triggered.

```python
from menace.evolution_scheduler import EvolutionScheduler

scheduler = EvolutionScheduler(
    orchestrator,
    my_data_bot,
    my_capital_bot,
    monitor=my_monitor,
    interval=30,
)
scheduler.start()
```

After each cycle the orchestrator automatically schedules A/B experiments for
any variants suggested by `ExperimentManager`. The results are stored in
`EvolutionHistoryDB` so that `EvolutionPredictor` can retrain and provide better
ROI forecasts for future cycles.

## Dependency notes

`EvolutionOrchestrator` runs evaluation steps through the `SelfImprovementEngine`,
which in turn calls `LearningEngine.evaluate()`. That evaluation uses
`sklearn.metrics` when available for holdout scoring. Install the evolution
extras to guarantee the dependency is present:

```bash
pip install .[evolution]
```

If scikit-learn is missing, the evaluation path logs a warning and falls back to
a lightweight accuracy calculation so orchestration can continue. This means
metrics are still produced, but the richer scikit-learn diagnostics are skipped.
