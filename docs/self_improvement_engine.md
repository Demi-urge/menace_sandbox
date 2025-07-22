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

Persisting cycle data across runs is possible by providing `state_path` when creating the engine. ROI deltas, an exponential moving average of those deltas (`roi_delta_ema`) and the timestamp of the last cycle are written to this JSON file and reloaded on startup. The smoothing factor for the EMA can be set with `roi_ema_alpha` (default `0.1`).

Each engine may use its own databases, event bus and automation pipeline allowing multiple bots to improve in parallel.

## Reinforcement-learning policy

`SelfImprovementEngine` can optionally be initialised with a `SelfImprovementPolicy`.
The policy learns from previous evolution cycles via Q‑learning and predicts
whether running another cycle is likely to increase ROI. `_should_trigger()`
consults this policy in addition to diagnostic checks and the predicted value
also scales the `energy` passed to the automation pipeline.

In addition to policy predictions, the engine tracks rolling averages of ROI
deltas. A short term mean and an exponential moving average influence the
energy allocated to each cycle. Positive averages boost the energy budget while
negative values reduce it.

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

## Synergy Weight Learners

The engine keeps a set of synergy weights that modulate how cross‑module
metrics influence policy updates. ``SynergyWeightLearner`` stores four weights
(``roi``, ``efficiency``, ``resilience`` and ``antifragility``) in a JSON file.
After each cycle ``_update_synergy_weights`` measures the rolling change of the
corresponding synergy metrics and calls ``SynergyWeightLearner.update``:

``synergy_roi`` → ``roi``

``synergy_efficiency`` → ``efficiency``

``synergy_resilience`` → ``resilience``

``synergy_antifragility`` → ``antifragility``

Each weight is nudged by ``lr * roi_delta * metric_delta`` and clamped between
``0`` and ``10``. Positive ROI deltas therefore reinforce metrics that improved
while negative deltas or worsening metrics lower their influence.

``DQNSynergyLearner`` extends this process with a small deep Q‑network. The
synergy deltas form the state and the ROI‑scaled change acts as the reward for
each action. Predicted Q‑values replace the manual gradient step so weight
updates follow the learned policy. Both learners persist their weights and any
model parameters to disk so progress carries over between runs.
