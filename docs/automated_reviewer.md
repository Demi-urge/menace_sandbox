# Automated Reviewer

`AutomatedReviewer` audits bots and escalates critical issues using LLM
prompts.

```python
from automated_reviewer import AutomatedReviewer
from vector_service.context_builder import ContextBuilder

builder = ContextBuilder("bots.db", "code.db", "errors.db", "workflows.db")
reviewer = AutomatedReviewer(context_builder=builder)
reviewer.handle({"bot_id": "1", "severity": "critical"})
```

The builder queries `bots.db`, `code.db`, `errors.db` and `workflows.db` and
compresses related snippets before embedding them in prompts.  When validation
fails, run `builder.validate()` to check that all database paths are reachable.
