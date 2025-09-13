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
