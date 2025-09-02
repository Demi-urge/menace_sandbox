# Prompt Chunking and Summary Cache

Large source files can exceed the token limits accepted by language models. The
`prompt_chunking` module splits code into manageable pieces and caches
summaries for reuse.

## Chunking behaviour

`chunk_code(path, token_limit)` walks the top level of a module and produces
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
