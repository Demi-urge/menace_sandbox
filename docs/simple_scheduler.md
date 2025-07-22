# Simple Scheduler

`_SimpleScheduler` provides a lightweight in-process scheduler used when
APScheduler is unavailable. Jobs are persisted to `scheduler_state.json` so they
survive restarts.

```python
from menace.cross_model_scheduler import _SimpleScheduler

sched = _SimpleScheduler()

def collect_metrics():
    ...

sched.add_job(collect_metrics, interval=60, id="metrics")
...
sched.shutdown()
```

Each job may specify a retry delay, maximum retries and a grace period for
misfires. The scheduler automatically restores all jobs from the state file on
startup.
