# Social Posting Scheduler

`Scheduler` in `clipped.scheduler` picks the best clip for each social account.
It loads clip metadata, account lists and topic definitions then ranks clips
using a simple moving average of views.  The top clip per account is returned
and the posting history is written to disk.

```python
from menace.clipped.scheduler import Scheduler

sched = Scheduler(
    clips_file="clips.json",
    topics_file="clip_topics.json",
    accounts_file="accounts.json",
    history_file="history.json",
)
results = sched.run()
```

Each clip can track how often it has been scheduled via the `scheduled`
field.  When `run()` completes, a JSON file with the chosen schedule
is written to `history_file`.
