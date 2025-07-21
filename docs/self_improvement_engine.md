# Self-Improvement Engine

`SelfImprovementEngine` runs a model automation pipeline whenever metrics signal the need for an update. The engine can target any bot by supplying a `bot_name` and pipeline instance.

```python
from menace.self_improvement_engine import SelfImprovementEngine, ImprovementEngineRegistry
from menace.model_automation_pipeline import ModelAutomationPipeline

engine = SelfImprovementEngine(bot_name="alpha",
                               pipeline=ModelAutomationPipeline())
registry = ImprovementEngineRegistry()
registry.register_engine("alpha", engine)

# Execute one cycle for all registered engines
registry.run_all_cycles()
```

When a `SelfCodingEngine` is supplied, the engine may patch helper code before running the automation pipeline. See [self_coding_engine.md](self_coding_engine.md) for more information.

Persisting cycle data across runs is possible by providing `state_path` when creating the engine. ROI deltas and the timestamp of the last cycle are written to this JSON file and reloaded on startup.

Each engine may use its own databases, event bus and automation pipeline allowing multiple bots to improve in parallel.

## Reinforcement-learning policy

`SelfImprovementEngine` can optionally be initialised with a `SelfImprovementPolicy`.
The policy learns from previous evolution cycles via Q‑learning and predicts
whether running another cycle is likely to increase ROI. `_should_trigger()`
consults this policy in addition to diagnostic checks and the predicted value
also scales the `energy` passed to the automation pipeline.

Recent sandbox runs record ROI deltas per module via ``sandbox_runner._SandboxMetaLogger``.
Passing the logger to `SelfImprovementEngine` exposes `rankings()` and
`diminishing()` so module trends influence the policy state. Module names are
hashed and persisted to keep the Q‑learning state space bounded.

## Automatic scaling

`ImprovementEngineRegistry.autoscale()` can create or remove engines
automatically based on available resources and the long‑term ROI trend.
Provide it with a `CapitalManagementBot`, a `DataBot` and a factory function
that returns a new `SelfImprovementEngine` for a given name:

```python
def factory(name: str) -> SelfImprovementEngine:
    return SelfImprovementEngine(bot_name=name,
                                 pipeline=ModelAutomationPipeline())

registry = ImprovementEngineRegistry()
registry.register_engine("alpha", factory("alpha"))

# Adjust the number of engines according to ROI and energy
registry.autoscale(capital_bot=my_capital_bot,
                   data_bot=my_data_bot,
                   factory=factory,
                   cost_per_engine=0.05,
                   approval_callback=lambda: True)
```

When the energy score exceeds ``create_energy`` (default ``0.8``) and the ROI
trend is above ``roi_threshold`` the registry ensures projected ROI covers the
``cost_per_engine`` before adding a new one. Optionally ``approval_callback``
can gate risky expansions. If resources dwindle or ROI falls below the
threshold, engines are removed down to ``min_engines``.
