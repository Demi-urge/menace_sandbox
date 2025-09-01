# LLM interface

The `llm_interface` module supplies a tiny interface for working with language models. It defines
the `Prompt` and `LLMResult` dataclasses together with the `LLMClient` base class which exposes a
single `generate` method. `LLMResult` captures both the raw text produced by the model and an
optional parsed representation.

## Basic usage

```python
from llm_interface import Prompt, LLMResult, LLMClient

class EchoClient(LLMClient):
    def generate(self, prompt: Prompt) -> LLMResult:
        # ``parsed`` may contain structured data such as a JSON dict.  It is
        # optional and defaults to ``None``.
        return LLMResult(text=prompt.text.upper())

client = EchoClient()
result = client.generate(Prompt(text="hello"))
print(result.text)
```

## Retry and backoff

Use `retry_utils.with_retry` to add exponential backoff around LLM calls:

```python
from retry_utils import with_retry

response = with_retry(lambda: client.generate(Prompt("hi")), attempts=3, delay=1.0)
```

## Prompt logging

`PromptDB` can store prompts and responses for later inspection. An in‑memory router is
sufficient for tests or temporary sessions:

```python
import sqlite3
from prompt_db import PromptDB

class MemoryRouter:
    def __init__(self):
        self.conn = sqlite3.connect(":memory:")
    def get_connection(self, table_name: str, operation: str = "write"):
        return self.conn

memory_db = PromptDB(model="demo", router=MemoryRouter())
memory_db.log(Prompt("hi", outcome_tags=["tag"], vector_confidences=[0.9]), LLMResult(text="ok", parsed={}))
```

## Fallback routing

`LLMRouter` chooses between two clients and falls back if the primary fails:

```python
from llm_router import LLMRouter

router = LLMRouter(remote=remote_client, local=backup_client, size_threshold=500)
result = router.generate(Prompt("data"))
```

## Extending

To support a new backend, implement `LLMClient.generate` and optionally expose a factory
function. The router and `client_from_settings` helper can then compose the new client
alongside existing ones.
