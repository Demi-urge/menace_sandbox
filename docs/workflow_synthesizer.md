# Workflow Synthesizer

`WorkflowSynthesizer` expands a seed module by blending structural signals from
`ModuleSynergyGrapher` with semantic intent search. It inspects module inputs
and outputs to resolve dependency order and emits small workflow candidates.

## Design

The synthesizer queries `ModuleSynergyGrapher` for related modules and scores
them against a text description. Candidates are explored breadth‑first and
pruned by synergy weight and semantic similarity. Internal state tracks which
modules have been selected and the values each step provides.

## I/O analysis

Each candidate module is introspected to determine its expected inputs and the
values it returns. Function signatures supply argument names while docstring or
return annotations describe outputs. The synthesizer aggregates this metadata so
later steps can consume values produced earlier in the workflow.

## Dependency resolution

When assembling a workflow the synthesizer orders modules so that every required
input is satisfied by a previous output. Any unresolved parameters are surfaced
in the step description for manual filling or further synthesis. This ensures
generated workflows can be executed or converted into specifications without
missing dependencies.

## Examples

### Programmatic usage

```python
from workflow_synthesizer import WorkflowSynthesizer

synth = WorkflowSynthesizer()
steps = synth.synthesize(start_module="module_a", problem="summarise data")
for step in steps:
    print(step["module"], step["args"], "->", step["provides"])
```

### Saving and CLI

```bash
# Generate candidate workflows and interactively save a spec
python workflow_synthesizer_cli.py --start module_a --out my.workflow.json
```

The `generate_workflows` method returns multiple candidates and `save_workflow`
can write a `.workflow.json` compatible with `WorkflowDB` for later reuse.

## CLI options

`workflow_synthesizer_cli.py` offers a small command line interface:

* `--start` – starting module name or a free text problem description. If a
  Python module with this name exists the synthesizer performs structural
  expansion; otherwise it searches by intent.
* `--out` – destination path for the generated workflow specification. The
  tool interactively confirms each suggested step before writing the file.

## Sandbox integration

Generated specifications are ordinary `.workflow.json` files. They can be
loaded into the sandbox's `WorkflowDB` or executed via existing workflow tools.
By default the synthesizer reads the synergy graph from
`sandbox_data/module_synergy_graph.json` and can optionally leverage an
`IntentDB` when available, allowing new workflows to plug into the rest of the
sandbox infrastructure without additional configuration.

