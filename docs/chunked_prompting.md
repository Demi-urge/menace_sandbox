# Chunked Prompting

Large files or prompts can easily exceed the token limits accepted by language models. Chunked prompting breaks content into smaller pieces and summarises each chunk so that the system can work with arbitrarily large inputs while staying within model limits.

## Configuration

`SandboxSettings` exposes two fields that control this behaviour:

- `chunk_token_threshold` – maximum tokens allowed per chunk before summarisation occurs.
- `chunk_summary_cache_dir` – directory where cached summaries are stored.

Both values can be overridden with the `CHUNK_TOKEN_THRESHOLD` and `CHUNK_SUMMARY_CACHE_DIR` environment variables. The cache directory defaults to `chunk_summary_cache/` inside the repository.

## Cache maintenance

The summary cache accelerates repeated runs by reusing existing summaries. It is safe to delete the cache when disk space becomes an issue or you want to force fresh summaries:

```
rm -rf chunk_summary_cache
```

A new directory is created automatically on demand. Periodic cleanup prevents stale summaries from accumulating.

