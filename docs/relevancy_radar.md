# Relevancy Radar

The relevancy radar suite keeps the repository lean by tracking how modules are
used and recommending maintenance actions. It combines three components:
`RelevancyRadar` gathers usage statistics, `RelevancyRadarService` evaluates
those findings on a schedule and `ModuleRetirementService` performs any
resulting cleanup.

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
