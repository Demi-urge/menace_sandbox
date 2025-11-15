# internalize_coding_bot

`internalize_coding_bot` wires a coding bot into the self‑coding system by:

- creating a `SelfCodingManager` for the bot,
- registering ROI/error thresholds with `BotRegistry`,
- registering the bot with `EvolutionOrchestrator` and subscribing to `degradation:detected` events.

## Example

```python
from menace.self_coding_manager import internalize_coding_bot
from menace.self_coding_engine import SelfCodingEngine
from menace.model_automation_pipeline import ModelAutomationPipeline
from menace.data_bot import DataBot
from menace.bot_registry import BotRegistry
from menace.evolution_orchestrator import EvolutionOrchestrator

engine = SelfCodingEngine(...)
pipeline = ModelAutomationPipeline(...)
data_bot = DataBot()
registry = BotRegistry()
orchestrator = EvolutionOrchestrator(data_bot, ...)

manager = internalize_coding_bot(
    "example-bot",
    engine,
    pipeline,
    data_bot=data_bot,
    bot_registry=registry,
    evolution_orchestrator=orchestrator,
    roi_threshold=-0.1,
    error_threshold=0.2,
)
```

Each coding bot should invoke this helper during initialisation to ensure
recursive integrity and automatic patch cycles.

## Constructing pipelines ahead of bootstrap

Maintenance scripts often build a `ModelAutomationPipeline` before
`internalize_coding_bot` starts the real bootstrap sequence.  Helper bootstrap
will otherwise observe `manager=None` and attempt to call `_bootstrap_manager`
again which emits "re-entrant initialisation depth" warnings and returns a
disabled sentinel.  To avoid that recursion, always pass a manager (either a
real `SelfCodingManager` or the fallback stub provided by
`coding_bot_interface.fallback_helper_manager`) into the pipeline constructor
and keep the helper override active while the pipeline wires up:

```python
from menace_sandbox.coding_bot_interface import (
    fallback_helper_manager,
    prepare_pipeline_for_bootstrap,
)
from menace_sandbox.model_automation_pipeline import ModelAutomationPipeline

builder = create_context_builder()
registry = BotRegistry()
data_bot = DataBot(start_server=False)

with fallback_helper_manager(bot_registry=registry, data_bot=data_bot) as manager:
    pipeline, promote = prepare_pipeline_for_bootstrap(
        pipeline_cls=ModelAutomationPipeline,
        context_builder=builder,
        bot_registry=registry,
        data_bot=data_bot,
        bootstrap_runtime_manager=manager,
        manager=manager,
    )

manager = internalize_coding_bot(
    "example-bot",
    engine,
    pipeline,
    bot_registry=registry,
    data_bot=data_bot,
)
promote(manager)
```

The helper exposes a `_DisabledSelfCodingManager` placeholder so nested helpers
observe a consistent manager and stay out of `_bootstrap_manager` until the real
manager has been registered.

## Pre-commit check

The pre-commit configuration ships with a `check-self-coding-registration`
hook powered by `tools/check_self_coding_registration.py`. The hook scans each
Python module for top-level classes or functions whose names end in `Bot` and
verifies they either call `internalize_coding_bot` or use the
`@self_coding_managed` decorator. Commits introducing unmanaged bots fail with a
non-zero status.

To satisfy the hook:

1. Call `internalize_coding_bot` during module initialisation, **or**
2. Decorate the bot with `@self_coding_managed` providing
   `bot_registry`, `data_bot`, and `SelfCodingManager`.

This ensures all exported bots participate in the self-coding lifecycle.

For a comprehensive audit of the repository, run
`tools/check_self_coding_decorator.py` which walks all subpackages and fails
when any class ending in `Bot` lacks the `@self_coding_managed` decorator.
