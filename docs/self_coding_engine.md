# Self-Coding Engine

`SelfCodingEngine` automatically builds helper functions from existing code snippets and appends them to a file. It can query an LLM to write helpers, run linting and tests and revert changes when performance drops.

## Usage

```python
from dynamic_path_router import resolve_path
from menace.self_coding_engine import SelfCodingEngine
from menace.code_database import CodeDB
from menace.menace_memory_manager import MenaceMemoryManager
from vector_service.context_builder import ContextBuilder

# SelfCodingEngine requires an explicit ContextBuilder instance
builder = ContextBuilder("bots.db", "code.db", "errors.db", "workflows.db")
builder.refresh_db_weights()
engine = SelfCodingEngine(
    CodeDB(resolve_path('code.db')),
    MenaceMemoryManager(resolve_path('mem.db')),
    context_builder=builder,
)
engine.apply_patch(resolve_path('utils.py'), 'normalize text')
```

Set `MENACE_ROOT` or `SANDBOX_REPO_PATH` to point the resolver at a different
clone. For multi-root setups specify `MENACE_ROOTS` or `SANDBOX_REPO_PATHS` and
pass `repo_hint` to `resolve_path` when targeting a specific checkout:

```bash
MENACE_ROOTS="/repo/main:/repo/alt" python - <<'PY'
from dynamic_path_router import resolve_path
print(resolve_path('code.db', repo_hint='/repo/alt'))
PY
```

- **suggest_snippets** – fetch related `CodeRecord` objects from `CodeDB`.
- **generate_helper** – uses `LLMClient` to produce a helper based on the description and snippet context.
- **apply_patch** – append the helper, run CI checks and revert when ROI, error or complexity deltas fall below historical baselines.
- **Patch metrics** – `PatchHistoryDB` stores ROI delta and error counts for each patch.
- **Forecasts** – `TrendPredictor` estimates ROI and error trends for rollback decisions.
- **Safety** – integrates with `SafetyMonitor` and records patch history so that problematic patches can be rolled back.
- **refactor_worst_bot** – automatically refactors the bot with the highest error rate using recent metrics.
- **Automatic Git sync** – when a patch is kept, `./sync_git.sh` commits and pushes the changes.

## GPTMemory Workflow

`SelfCodingEngine` accepts a `gpt_memory` (or legacy `gpt_memory_manager`)
parameter.  When provided, every prompt/response pair exchanged with the LLM is
recorded so future patches can reuse relevant context.

```python
from gpt_memory import GPTMemoryManager
from vector_service.context_builder import ContextBuilder

builder = ContextBuilder("bots.db", "code.db", "errors.db", "workflows.db")
builder.refresh_db_weights()
engine = SelfCodingEngine(
    CodeDB("code.db"),
    MenaceMemoryManager("mem.db"),
    gpt_memory=GPTMemoryManager("helpers.db"),
    context_builder=builder,
)
engine.gpt_memory.log_interaction("write sort helper", "generated code", tags=["bugfix"])
```

Key options mirror those in `GPTMemoryManager`:

- `db_path` – path to the SQLite file; use `":memory:"` for ephemeral
  sessions.
- `embedder` – optional `SentenceTransformer` enabling semantic search through
  `get_similar_entries()`.

See [gpt_memory.md](gpt_memory.md) for more examples and configuration details.

## Ephemeral Test Environments

`SelfCodingEngine` evaluates patches inside isolated environments. The helper
`create_ephemeral_env` clones the working directory into a temporary virtual
environment (or Docker container) and installs dependencies from
`requirements.txt` before running tests.

```python
from pathlib import Path
from sandbox_runner.environment import create_ephemeral_env

with create_ephemeral_env(Path(".")) as (repo, run):
    (repo / "test_example.py").write_text("def test_ok():\n    assert True\n")
    run(["pytest", "-q"], check=True)
```

Startup time and installation failures are logged via the sandbox logging
utilities so that callers can track environment issues.

## Prompt chunking and caching

Large files are summarised in smaller chunks when their token count exceeds
`prompt_chunk_token_threshold`. Chunk summaries are cached under
`chunk_summary_cache_dir` to speed up subsequent runs. Both settings are
configurable through `SandboxSettings` or environment variables
(`PROMPT_CHUNK_TOKEN_THRESHOLD` and `CHUNK_SUMMARY_CACHE_DIR`, with
`PROMPT_CHUNK_CACHE_DIR` accepted for backward compatibility).

## Retry and fallback

LLM calls are retried with backoff.  The delay between attempts is controlled by
`SandboxSettings.codex_retry_delays` (environment variable
`CODEX_RETRY_DELAYS`) and defaults to `[2, 5, 10]` seconds.  When all retries
fail the prompt is simplified – examples are trimmed and system text is removed
– before one final attempt.

To override the schedule set `CODEX_RETRY_DELAYS` in the environment to a
comma‑separated list or JSON array (e.g. `"1,2,4"` or `[1,2,4]`).

If no code is produced, `codex_fallback_handler.handle` either queues the prompt
or reroutes it to a lower‑cost model.  The function always returns an
`LLMResult`; when rerouting fails, the result has an empty `text` field and the
failure reason in `result.raw`.  Inspect `result.text` for alternate completions
and `result.raw` for provider metadata.  Select the behaviour via
`CODEX_FALLBACK_STRATEGY` (`"queue"` or `"reroute"`; default) and specify the
alternate model with `CODEX_FALLBACK_MODEL` (defaults to `gpt-3.5-turbo`).
Queued prompts are written to `CODEX_RETRY_QUEUE` (`codex_retry_queue_path`).

Queued requests provide an empty `LLMResult` so the patch loop carries on, while
rerouted completions are marked as degraded but still allow the cycle to
proceed.  These fallbacks keep the self‑coding loop running even when the
primary model is unavailable.


