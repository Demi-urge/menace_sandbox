# Active Learning

`CurriculumBuilder` analyses telemetry via `ErrorBot.summarize_telemetry()` and
creates a curriculum of frequent failure types. Items whose occurrence count
exceeds the configured threshold are published on the `"curriculum:new"` topic of
the event bus.

`SelfLearningCoordinator` subscribes to this topic. Every curriculum entry is
converted to a `PathwayRecord` with a failing outcome and passed to the
registered learning engines via `partial_train()` so models adapt to common
errors without waiting for a full training cycle.

```python
from menace.unified_event_bus import UnifiedEventBus
from menace.error_bot import ErrorBot
from menace.curriculum_builder import CurriculumBuilder
from menace.self_learning_coordinator import SelfLearningCoordinator
from vector_service.context_builder import ContextBuilder

bus = UnifiedEventBus()
ctx = ContextBuilder("bots.db", "code.db", "errors.db", "workflows.db")
err_bot = ErrorBot(context_builder=ctx)
builder = CurriculumBuilder(err_bot, bus, threshold=5)
coord = SelfLearningCoordinator(bus, curriculum_builder=builder)
coord.start()
builder.publish()
```

With this integration new high-frequency errors automatically generate training
data and trigger incremental updates.
