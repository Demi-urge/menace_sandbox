# LLM interface

The `llm_interface` module supplies a tiny protocol for working with language models. It defines
`Prompt`, `LLMResult` and the `LLMClient` protocol which requires a single `generate` method.

## Basic usage

```python
from llm_interface import Prompt, LLMResult, LLMClient

class EchoClient(LLMClient):
    def generate(self, prompt: Prompt) -> LLMResult:
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
memory_db.log_prompt(Prompt("hi"), LLMResult(text="ok"), ["tag"], 0.9)
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
