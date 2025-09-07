# Prompt Engine

The `prompt_engine` module assembles prompts from historical patch examples.
It queries the retrieval layer for relevant patches, ranks the results and
formats them into reusable snippets.

## Usage

```python
from prompt_engine import PromptEngine
from vector_service.context_builder import ContextBuilder

builder = ContextBuilder("bots.db", "code.db", "errors.db", "workflows.db")
prompt = PromptEngine.construct_prompt(
    "Fix the failing parser",
    retry_trace="Traceback: ValueError",
    top_n=3,
    context_builder=builder,
)
print(prompt)
```

All prompt‑constructing bots (e.g., `SelfCodingEngine`, `QuickFixEngine`) must
provide a `ContextBuilder` via the `context_builder` argument so prompts include
vector context.

A successful lookup yields a prompt such as:

```
Given the following pattern, fix the failing parser

Given the following pattern:
Code summary: handle edge case
Diff summary: adjust parsing
Outcome: works (tests passed)
```

## Configuration

* `top_n` – number of patch examples to retrieve (default: `5`).
* `DEFAULT_TEMPLATE` – fallback text when no snippets are available.
* `CONFIDENCE_THRESHOLD` – minimum confidence before using the fallback
  template.
* `retry_trace` – when provided, the prompt includes:

  ```
  Previous failure:
  <trace>
  Please attempt a different solution.
  ```
* `token_threshold` – Maximum number of tokens permitted for the assembled prompt.
  Text exceeding this limit is trimmed and long files are summarised when
  included as context.
* `chunk_token_threshold` – Token limit for individual code chunks when large files
  are summarised (defaults to `PROMPT_CHUNK_TOKEN_THRESHOLD`).
* `prompt_chunk_cache_dir` – Directory used to store cached summaries of
  chunked code (`PROMPT_CHUNK_CACHE_DIR`).
* `success_header` and `failure_header` – control the section titles for
  successful and failing examples.  The defaults are
  `"Given the following pattern:"` and
  `"Avoid {summary} because it caused {outcome}:"`.

### Custom headers

Codex-style prompts can be produced by overriding these headers:

```python
engine = PromptEngine(
    retriever=my_retriever,
    success_header="Correct example:",
    failure_header="Incorrect example:",
)
```

Which yields output of the form:

```
Correct example:
<successful snippet>

Incorrect example:
<failing snippet>
```

## Learning from history

`PromptMemoryTrainer` analyses previous prompts and patch outcomes to infer
effective formatting.  The trainer now detects additional cues such as the
presence of fenced code blocks, bullet lists and separate `System`/`User`
sections.  Success rates for each observed style are weighted by the ROI or
patch complexity improvement recorded for the corresponding patch, allowing
the prompt engine to favour styles linked to more impactful changes.
