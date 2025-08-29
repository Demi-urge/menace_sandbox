# Relevancy Radar

The relevancy radar suite keeps the repository lean by tracking how modules are
used and recommending maintenance actions. It combines three components:
`RelevancyRadar` gathers usage statistics, `RelevancyRadarService` evaluates
those findings on a schedule and `ModuleRetirementService` performs any
resulting cleanup.

## Installation

`RelevancyRadar` uses `networkx` for dependency analysis. Install it with:

```bash
pip install networkx
```

## Components and interaction

- **`RelevancyRadar`** collects import and execution counts, records optional
  impact scores and evaluates modules against threshold values. Results are
  persisted to `sandbox_data/relevancy_metrics.json`.
- **`RelevancyRadarService`** periodically invokes the radar. It merges live
  usage statistics and ROI deltas, updates Prometheus metrics and forwards any
  flags to the retirement service.
- **`ModuleRetirementService`** performs the requested action. Unused modules
  are archived, low‑value modules are compressed via the quick fix engine and
  replaceable modules receive a generated patch proposal.

### Recording Output Impact

Use `map_module_identifier` with an ROI delta to register how much a module
changed the run's return on investment. When that module influences the final
output, call `record_output_impact` so the radar can attribute the delta.

```python
from pathlib import Path
from sandbox_runner.cycle import map_module_identifier
from relevancy_radar import record_output_impact, radar

repo = Path("/workspace/menace_sandbox")
module_id = map_module_identifier("analytics/stats.py", repo, 0.8)

# when analytics/stats contributes to the result
record_output_impact(module_id, 0.8)
```

### Decorator-based tracking

Wrap high-level functions with `@radar.track` to automatically record usage and
forward any ROI deltas to the radar. The decorator inspects return values and
``impact``/``roi_delta`` arguments to attribute output impact.

```python
from relevancy_radar import radar

@radar.track
def run_cycle() -> float:
    # ... perform work and return ROI delta ...
    return 1.2
```

### Example flows

**Retirement**
1. `RelevancyRadar` notices that `old_helper` has no recorded activity.
2. `RelevancyRadarService` emits a `retire` flag.
3. `ModuleRetirementService` moves `old_helper.py` into
   `sandbox_data/retired_modules/` when no dependents are found.

**Compression**
1. `RelevancyRadar` scores `slow_algo` below the compression threshold.
2. `RelevancyRadarService` flags it for `compress`.
3. `ModuleRetirementService` calls `generate_patch` to produce a smaller
   version.

**Replacement**
1. `RelevancyRadar` identifies `legacy_api` as rarely used but still necessary.
2. `RelevancyRadarService` labels it `replace`.
3. `ModuleRetirementService` requests a patch from the quick fix tooling and
   logs the patch identifier.

## CLI usage

```
python relevancy_radar_cli.py --retire old_mod --compress slow_mod --replace alt_mod
```

### Dependency-aware evaluation

Pass `--final` to run `evaluate_final_contribution` before listing modules.
This performs a dependency-aware analysis that walks the import graph and
attributes usage to reachable modules. Modules outside the core dependency
chains are flagged for retirement, while those with low combined import,
execution and impact scores are annotated for compression or replacement.

The evaluation runs prior to displaying metrics, ensuring that any dependency
adjustments are reflected in the output.

## Display options

- `--threshold`: score below which modules are shown (default `5`).
- `--show-impact`: include cumulative impact scores in the listing.
- `--final`: run the dependency-aware evaluation step described above.
