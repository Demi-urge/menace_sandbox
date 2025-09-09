# Prompt Chunking and Summary Cache

Large source files can exceed the token limits accepted by language models. The
consolidated `chunking` module splits code into manageable pieces and caches
summaries for reuse. It uses `tiktoken` when available for accurate token counts
and falls back to simple word splitting otherwise. Summaries are generated via a
tiny diff model, an LLM client, or finally by heuristic line extraction if both
helpers are unavailable.

## Chunking behaviour

`chunk_file(path, token_limit)` walks the top level of a module and produces
chunks whose token count stays below `token_limit`. Token counts are estimated
with the best available tokenizer and gracefully degrade to a word split when
needed.

### Adjusting token thresholds

The `prompt_chunk_token_threshold` setting controls how many tokens a chunk may
contain before it is split. Adjust it via
`PROMPT_CHUNK_TOKEN_THRESHOLD` or the corresponding field in
`SandboxSettings` to balance fidelity against prompt size.

## Cache semantics

`get_chunk_summaries(path, token_limit, context_builder)` uses
`chunk_summary_cache.ChunkSummaryCache` to persist summaries for each file. The
cache tracks the file's content hash and automatically invalidates entries when
the source changes. Subsequent calls for unchanged files reuse the stored
summaries, making repeated prompt generation faster without manual file-hash
bookkeeping.

### Managing the summary cache

Cached summaries live under `chunk_summary_cache_dir` (environment variable
`CHUNK_SUMMARY_CACHE_DIR`, with `PROMPT_CHUNK_CACHE_DIR` accepted for backward
compatibility). Deleting the directory forces regeneration and is safe when disk
space becomes an issue.

## Developer notes

To target other languages or LLMs you can drop in a custom implementation:

```python
from my_llm import client

def summarize_code(code: str, context_builder) -> str:
    ctx = context_builder.build(code)
    return client.summarize(f"{code}\n\nContext:\n{ctx}", language="rust")
```

Monkeypatching `chunking.summarize_code` or providing an alternative
`code_summarizer.summarize_code` allows the cache to store summaries from any
backend. Summaries should remain short and deterministic so cache hits stay
reliable across runs.
