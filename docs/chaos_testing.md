# Chaos Testing

`ChaosScheduler` works with `ChaosTester` to randomly kill processes or suspend threads of core bots. It can also corrupt disk files and temporarily partition the network to simulate more destructive faults. Configure it with the targets to disrupt and an optional `Watchdog` instance.

```python
from menace.chaos_scheduler import ChaosScheduler
from menace.watchdog import Watchdog
from vector_service.context_builder import ContextBuilder

builder = ContextBuilder("bots.db", "code.db", "errors.db", "workflows.db")
scheduler = ChaosScheduler(
    processes=[proc],
    bots=[bot],
    disk_paths=["/tmp/data.db"],
    hosts=["db", "cache"],
    watchdog=Watchdog(..., context_builder=builder),
)
scheduler.start()
```
`Watchdog` receives the builder via the `context_builder` argument.
When a fault is injected the scheduler calls `Watchdog.record_fault()` which logs the event and checks if the provided bot recovered via `ChaosTester.validate_recovery()`. Fault history is stored in-memory and included when operators gather diagnostics.

Failed workflows can also be replayed once issues are resolved. Pass the `workflow` name to
`Watchdog.record_fault()` and later call `Watchdog.validate_workflows()` with a
`ReplayValidator`. The results are forwarded to the orchestrator which updates
confidence metrics for each workflow.
