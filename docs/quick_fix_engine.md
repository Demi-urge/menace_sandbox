# Quick Fix Engine

`QuickFixEngine` suggests patches for failing tests or runtime errors.

```python
from quick_fix_engine import QuickFixEngine, generate_patch
from error_bot import ErrorDB
from self_coding_manager import SelfCodingManager
from vector_service.context_builder import ContextBuilder

builder = ContextBuilder("bots.db", "code.db", "errors.db", "workflows.db")
engine = QuickFixEngine(ErrorDB(), SelfCodingManager(), context_builder=builder)
generate_patch("sandbox_runner", context_builder=builder)
```

The builder queries `bots.db`, `code.db`, `errors.db` and `workflows.db` and
compresses snippets before including them in prompts.  If validation fails,
`builder.validate()` reports missing or unreadable database files.
