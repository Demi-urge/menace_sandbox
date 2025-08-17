# Service layer

The sandbox exposes lightweight service classes that wrap background tasks and
long‑running loops.  Each service provides a minimal API and supports dependency
injection for easy testing.

Common patterns:

* `run_once()` methods execute the core logic a single time.
* `run_continuous(interval, stop_event)` starts a background thread that
  repeatedly invokes the logic until the supplied ``threading.Event`` is set.
* Dependencies such as schedulers, databases or planners can be passed in at
  construction time, allowing other modules to plug in custom implementations or
  stubs in tests.
* Errors are logged and many services retry internally so callers do not need to
  implement their own loops.

## Integrating with the service layer

Instantiate the service with any required collaborators and call
``run_continuous`` to start the background task.  Provide a ``threading.Event``
when you need to stop it:

```python
import threading
from microtrend_service import MicrotrendService

svc = MicrotrendService()
stop = threading.Event()
svc.run_continuous(interval=3600, stop_event=stop)
# ... later ...
stop.set()
```

This pattern is shared by other services such as ``DebugLoopService`` for
archiving crash traces, ``ChaosMonitoringService`` for automated rollbacks and
``SelfEvaluationService`` for cloning high‑performing workflows.
