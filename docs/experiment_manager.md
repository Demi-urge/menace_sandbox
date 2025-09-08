# Experiment Manager

`ExperimentManager` allows simple A/B testing of alternative bot implementations.
It runs each variant through the `ModelAutomationPipeline` and collects ROI
metrics via `DataBot` and `CapitalManagementBot`. Statistical tests are performed
using `scipy.stats` and the best-performing variant is only promoted when the
p-value is below the configured threshold.

Example:
```python
from menace.experiment_manager import ExperimentManager
from vector_service.context_builder import ContextBuilder

builder = ContextBuilder("bots.db", "code.db", "errors.db", "workflows.db")
manager = ExperimentManager(data_bot, capital_bot, context_builder=builder)
results = await manager.run_experiments(["bot_v1", "bot_v2"], energy=2)
best = manager.best_variant(results)

# Automatically pick variants suggested by the prediction manager
best = await manager.run_suggested_experiments("core_bot")
```

Each `ExperimentResult` now includes the sample size used for ROI calculation
and the variance of those samples which are used when running statistical
comparisons.

