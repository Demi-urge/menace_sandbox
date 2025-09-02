# Chunking and Summary Cache

Large source files can exceed the token limits accepted by language models. The
`chunking` module splits code into manageable pieces and caches summaries for
reuse.

## Chunking behaviour

`chunk_file(path, token_limit)` walks the top level of a module and produces
chunks whose token count stays below `token_limit`. Token counts are estimated
using the best available tokenizer and fall back to a simple word split when
none is available.

## Cache semantics

`get_chunk_summaries(path, token_limit)` hashes each chunk and stores a JSON
record containing the chunk and its summary inside the `chunk_summary_cache/`
folder. On subsequent runs the hash is used to look up an existing summary. If
it matches the current chunk the cached summary is reused; otherwise a new
summary is generated and written back to the cache. The cache makes repeated
prompt generation faster and now logs when cache hits or misses occur.

## Developer notes

The default `summarize_code` helper uses a tiny diff model and simply returns the
original code if the model is unavailable. To target other languages or LLMs you
can drop in a custom implementation:

```python
from my_llm import client

def summarize_code(code: str) -> str:
    return client.summarize(code, language="rust")
```

Monkeypatching `chunking.summarize_code` or providing an alternative
`code_summarizer.summarize_code` allows the cache to store summaries from any
backend. Summaries should remain short and deterministic so cache hits stay
reliable across runs.
