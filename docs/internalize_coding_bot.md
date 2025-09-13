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

## Pre-commit check

Repositories using the provided pre-commit configuration run
`tools/check_self_coding_registration.py` to ensure every module that exports a
class or function ending in `Bot` either calls `internalize_coding_bot` or uses
the `@self_coding_managed` decorator.  If a bot fails these requirements the
hook exits with a non-zero status and the commit is rejected.

To pass the check:

1. Call `internalize_coding_bot` during module initialisation, **or**
2. Decorate the bot with `@self_coding_managed`.

This guarantees all exported bots participate in the self-coding lifecycle.
