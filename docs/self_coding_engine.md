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
- **generate_helper** – uses `ChatGPTClient` to produce a helper based on the description and snippet context.
- **apply_patch** – append the helper, run CI checks and revert if ROI or error metrics drop beyond ``threshold``.
- **Patch metrics** – `PatchHistoryDB` stores ROI delta and error counts for each patch.
- **Forecasts** – `TrendPredictor` estimates ROI and error trends for rollback decisions.
- **Safety** – integrates with `SafetyMonitor` and records patch history so that problematic patches can be rolled back.
- **refactor_worst_bot** – automatically refactors the bot with the highest error rate using recent metrics.
- **Automatic Git sync** – when a patch is kept, `./sync_git.sh` commits and pushes the changes.


