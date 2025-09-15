# LLM interface

The `llm_interface` module supplies a tiny interface for working with language models. It defines
the `Prompt` and `LLMResult` dataclasses together with the `LLMClient` base class which exposes a
single `generate` method. `LLMResult` captures both the raw text produced by the model and an
optional parsed representation.  ``Prompt`` objects should be created via
``ContextBuilder.build_prompt`` or ``SelfCodingEngine.build_enriched_prompt``;
constructing them directly in application code is prohibited.

## Basic usage

```python
from llm_interface import Prompt, LLMResult, LLMClient
from vector_service.context_builder import ContextBuilder

class EchoClient(LLMClient):
    def generate(self, prompt: Prompt, *, context_builder: ContextBuilder) -> LLMResult:
        # ``parsed`` may contain structured data such as a JSON dict.  It is
        # optional and defaults to ``None``.
        return LLMResult(text=prompt.text.upper())

client = EchoClient()
builder = ContextBuilder()
prompt = builder.build_prompt("hello")
result = client.generate(prompt, context_builder=builder)
print(result.text)
```

## Retry and backoff

Use `retry_utils.with_retry` to add exponential backoff around LLM calls:

```python
from retry_utils import with_retry

response = with_retry(
    lambda: client.generate(builder.build_prompt("hi"), context_builder=builder),
    attempts=3,
    delay=1.0,
)
```

## Prompt logging

`PromptDB` can store prompts and responses for later inspection. An in‑memory router is
sufficient for tests or temporary sessions:

```python
import sqlite3
from prompt_db import PromptDB
from vector_service.context_builder import ContextBuilder

class MemoryRouter:
    def __init__(self):
        self.conn = sqlite3.connect(":memory:")
    def get_connection(self, table_name: str, operation: str = "write"):
        return self.conn

memory_db = PromptDB(model="demo", router=MemoryRouter())
builder = ContextBuilder()
prompt = builder.build_prompt("hi")
memory_db.log(prompt, LLMResult(text="ok", parsed={}))
```

## Fallback routing

`LLMRouter` chooses between two clients and falls back if the primary fails:

```python
from llm_router import LLMRouter
from vector_service.context_builder import ContextBuilder

router = LLMRouter(remote=remote_client, local=backup_client, size_threshold=500)
builder = ContextBuilder()
prompt = builder.build_prompt("data")
result = router.generate(prompt, context_builder=builder)
```

## Extending

To support a new backend, implement `LLMClient.generate` and optionally expose a factory
function. The router and `client_from_settings` helper can then compose the new client
alongside existing ones.
