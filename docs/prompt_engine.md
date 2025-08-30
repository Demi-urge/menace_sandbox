# Prompt Engine

The `prompt_engine` module assembles prompts from historical patch examples.
It queries the retrieval layer for relevant patches, ranks the results and
formats them into reusable snippets.

## Usage

```python
from prompt_engine import PromptEngine

prompt = PromptEngine.construct_prompt(
    "Fix the failing parser",
    retry_trace="Traceback: ValueError",
    top_n=3,
)
print(prompt)
```

A successful lookup yields a prompt such as:

```
Given the following pattern...
- Code summary: handle edge case
  Diff summary: adjust parsing
  Outcome: works (tests passed)
Previous attempt failed with Traceback: ValueError; seek alternative solution.
```

## Configuration

* `top_n` – number of patch examples to retrieve (default: `5`).
* `DEFAULT_TEMPLATE` – fallback text when no snippets are available.
* `CONFIDENCE_THRESHOLD` – minimum confidence before using the fallback
  template.
* `retry_trace` – when provided, the prompt includes:
  `Previous attempt failed with <trace>; seek alternative solution.`
