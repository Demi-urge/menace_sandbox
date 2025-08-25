# Workflow Synthesizer

## Purpose

`WorkflowSynthesizer` proposes small, ordered workflows by blending structural
signals from `ModuleSynergyGrapher` with optional intent search. The tool
inspects each candidate module's inputs and outputs, arranges them so
dependencies are satisfied and emits a list of steps that can be executed or
further refined.

## CLI usage

The standalone helper `workflow_synthesizer_cli.py` exposes a minimal command
line interface:

```bash
python workflow_synthesizer_cli.py --start module_a --out my.workflow.json
python workflow_synthesizer_cli.py --start "summarise data" --out summary.workflow.json
```

* `--start` – starting module name. When the module does not exist the value is
  treated as a free‑text problem description used for intent matching.
* `--problem` – additional description to bias intent search.
* `--max-depth` – limit traversal depth when exploring connected modules.
* `--out` – file or directory where generated workflows are written.

The command prints the candidate workflows to stdout and, when `--out` is
supplied, persists them to disk.

## Output format

Saved workflows always end with `.workflow.json` and contain an ordered list of
steps:

```json
{
  "steps": [
    {"module": "module_a", "inputs": [], "outputs": ["result"]},
    {"module": "module_b", "inputs": ["result"], "outputs": []}
  ]
}
```

The structure mirrors the format produced by `workflow_spec.to_spec`, allowing
conversion to YAML or insertion into `WorkflowDB`.

## Sandbox evaluation pipeline integration

Generated `.workflow.json` files plug directly into the sandbox evaluation
pipeline. Register the file with `task_handoff_bot.WorkflowDB` and the sandbox
runner and evaluation workers will schedule, execute and score the workflow like
any other entry. Results flow through `EvaluationHistoryDB` and surface on the
standard dashboards without additional configuration.

