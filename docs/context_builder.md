# Context Builder

`ContextBuilder` composes a compact context block by querying multiple databases through [`UniversalRetriever`](universal_retriever.md). It summarises records from the error, bot, workflow and code databases so language-model helpers receive the most relevant background.

## Usage

```python
from menace.context_builder import ContextBuilder
from menace.error_bot import ErrorDB
from menace.bot_database import BotDB
from menace.task_handoff_bot import WorkflowDB
from menace.code_database import CodeDB

err_db = ErrorDB()
bot_db = BotDB()
wf_db = WorkflowDB()
code_db = CodeDB()

builder = ContextBuilder(
    error_db=err_db,
    bot_db=bot_db,
    workflow_db=wf_db,
    code_db=code_db,
)
context = builder.build_context("upload failed", limit_per_type=5)
```

`build_context()` accepts a free-form query and returns a mapping:

```json
{
  "errors": [{"id": 1, "summary": "...", "metric": 0.8}],
  "bots":   [{"id": 2, "summary": "..."}],
  "workflows": [{"id": 3, "summary": "..."}],
  "code":   [{"id": 4, "summary": "..."}]
}
```

Each entry includes an identifier, a condensed summary and an optional metric such as ROI or resolution success.

## Integration

Code-generation helpers including `SelfCodingEngine`, `QuickFixEngine`, `BotDevelopmentBot` and `AutomatedReviewer` instantiate `ContextBuilder` automatically to enrich prompts with related history.

Control how many records are returned per type via the `limit_per_type` argument. Scoring weights come from the underlying `UniversalRetriever`; adjust parameters on `builder.retriever` (for example `link_multiplier`) to emphasise connectivity or tweak ranking.
