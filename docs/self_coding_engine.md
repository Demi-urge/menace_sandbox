# Self-Coding Engine

`SelfCodingEngine` automatically builds helper functions from existing code snippets and appends them to a file. It can query an LLM to write helpers, run linting and tests and revert changes when performance drops.

## Usage

```python
from pathlib import Path
from menace.self_coding_engine import SelfCodingEngine
from menace.code_database import CodeDB
from menace.menace_memory_manager import MenaceMemoryManager

engine = SelfCodingEngine(CodeDB("code.db"), MenaceMemoryManager("mem.db"))
engine.apply_patch(Path("utils.py"), "normalize text", threshold=0.5)
```

- **suggest_snippets** – fetch related `CodeRecord` objects from `CodeDB`.
- **generate_helper** – uses `LLMClient` to produce a helper based on the description and snippet context.
- **apply_patch** – append the helper, run CI checks and revert if ROI or error metrics drop beyond ``threshold``.
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

engine = SelfCodingEngine(
    CodeDB("code.db"),
    MenaceMemoryManager("mem.db"),
    gpt_memory=GPTMemoryManager("helpers.db"),
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


