# Prompt Evolution Memory

`PromptEvolutionMemory` persists structured records for prompt experiments so
that optimisation tools can learn from past executions. Each call to `log`
writes a JSON object to either a success or failure log, allowing subsequent
runs to analyse what formatting strategies yield the best return on investment
(ROI).

## Log format

Entries are appended to line-delimited JSON files. Successful executions are
written to `sandbox_data/prompt_success_log.jsonl` while failures go to
`sandbox_data/prompt_failure_log.jsonl`.

```json
{
  "timestamp": 1711631182,
  "prompt": {
    "system": "sys",
    "user": "generate code",
    "examples": ["sample" ]
  },
  "format": {"tone": "neutral", "headers": ["Intro"]},
  "result": {"stdout": "ok", "runtime": 1.2},
  "roi": {"roi_delta": 1.5, "coverage": 0.8}
}
```

Fields:

* `timestamp` – UNIX time of the execution.
* `prompt` – captured `system`/`user` text and `examples` list.
* `format` – metadata produced during prompt formatting (tone, section order,
  etc.).
* `result` – structured result of running the prompt (stdout, runtime,
  scores).
* `roi` – optional ROI metadata.  Numbers such as `roi_delta`, `coverage` or
  `runtime_improvement` can be stored and later used for weighting.

## ROI metadata

ROI fields describe the impact of a prompt.  Positive values represent an
improvement while negative values indicate a regression.  Multiple metrics can
be supplied and the optimizer can weight them during aggregation.  For example:

```python
logger.log(
    prompt,
    success=True,
    result={"stdout": "ok"},
    roi={"roi_delta": 2.3, "coverage": 0.9},
    format_meta={"tone": "upbeat"},
)
```

## Using the optimizer

`PromptOptimizer` consumes the success and failure logs to rank formatting
configurations.  It groups entries by structural features and computes success
rates and weighted ROI.  Suggested formats can then be fed back into a prompt
engine.

```python
from prompt_evolution_memory import PromptEvolutionMemory
from prompt_optimizer import PromptOptimizer
from prompt_engine import PromptEngine

logger = PromptEvolutionMemory()
# record prompt executions ...

optimizer = PromptOptimizer(
    logger.success_path,
    logger.failure_path,
    stats_path="prompt_optimizer_stats.json",
    weight_by="coverage",   # use the ``coverage`` ROI field for weighting
    roi_weight=1.2           # emphasise ROI in ranking
)

builder = ContextBuilder("bots.db", "code.db", "errors.db", "workflows.db")
engine = PromptEngine(context_builder=builder, retriever=my_retriever, optimizer=optimizer)
engine.build_prompt("add caching", context_builder=builder)
```

`PromptEngine` applies the optimizer's suggestions when formatting future
prompts, continually improving style and section ordering.

## Configuration options

### PromptEvolutionMemory

* `success_path` – destination for successful execution logs
  (default: `sandbox_data/prompt_success_log.jsonl`).
* `failure_path` – destination for failed execution logs
  (default: `sandbox_data/prompt_failure_log.jsonl`).

### PromptOptimizer

* `success_log` / `failure_log` – paths to the success and failure JSONL logs.
* `stats_path` – file used to persist aggregated statistics.
* `weight_by` – optional ROI field (`"coverage"`, `"runtime"`, etc.) used to
  weight ROI calculations.
* `roi_weight` – exponent applied to ROI when ranking formats; values above one
  favour ROI over raw success rate.

## Interpreting logs

Logs are standard JSONL and can be processed with any text tools.  The snippet
below prints the average ROI for successful prompts:

```python
import json
from pathlib import Path

success_log = Path("sandbox_data/prompt_success_log.jsonl")
entries = [json.loads(l) for l in success_log.read_text().splitlines() if l]
avg_roi = sum(e.get("roi", {}).get("roi_delta", 0) for e in entries) / len(entries)
print(f"average ROI: {avg_roi:.2f}")
```

## Integrating optimizer outputs

The optimizer returns ranked suggestions describing high-performing formats.
A simple integration might apply the top suggestion's metadata directly:

```python
suggestion = optimizer.suggest_format("prompt_engine", "build_prompt")[0]
print("Best tone:", suggestion["tone"])
print("Headers:", suggestion["structured_sections"])
```

These hints can be used to adjust prompt templates or to inform manual review.
