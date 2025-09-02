# Chunked Prompting

Large files or prompts can easily exceed the token limits accepted by language models. Chunked prompting breaks content into smaller pieces and summarises each chunk so that the system can work with arbitrarily large inputs while staying within model limits. The process relies on the consolidated `chunking` module described in [Prompt Chunking and Summary Cache](prompt_chunking.md), which falls back to simple heuristics when tokenisation or summarisation helpers are unavailable.

## Configuration

`SandboxSettings` exposes three related fields that control this behaviour:

- `prompt_chunk_token_threshold` – token limit used when splitting code for summarisation.
- `chunk_token_threshold` – maximum tokens allowed per chunk when constructing the final prompt.
- `chunk_summary_cache_dir` – directory where cached summaries are stored.

Override these via the `PROMPT_CHUNK_TOKEN_THRESHOLD`, `CHUNK_TOKEN_THRESHOLD` and `CHUNK_SUMMARY_CACHE_DIR` environment variables. The cache directory defaults to `chunk_summary_cache/` inside the repository, and the environment variable `PROMPT_CHUNK_CACHE_DIR` is still recognised for backward compatibility.

### Adjusting token thresholds

Lowering `prompt_chunk_token_threshold` produces smaller summaries but may lose detail. Increasing it allows larger chunks at the cost of bigger prompts. Tune the value to balance fidelity and model limits.

## Cache maintenance

The summary cache accelerates repeated runs by reusing existing summaries. Delete the cache when disk space becomes an issue or you want to force fresh summaries:

```
rm -rf chunk_summary_cache
```

The directory is recreated automatically on demand. Periodic cleanup prevents stale summaries from accumulating.

