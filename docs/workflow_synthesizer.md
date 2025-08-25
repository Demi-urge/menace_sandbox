# Workflow Synthesizer

`WorkflowSynthesizer` expands a seed module by blending structural signals from
`ModuleSynergyGrapher` with semantic intent search.  It inspects module inputs
and outputs to resolve dependency order and emits small workflow candidates.

## Quick start

```python
from workflow_synthesizer import WorkflowSynthesizer

synth = WorkflowSynthesizer()
steps = synth.synthesize(start_module="module_a", problem="summarise data")
for step in steps:
    print(step["module"], step["args"], "->", step["provides"])
```

The :meth:`WorkflowSynthesizer.synthesize` method returns a structured list of
steps where each entry describes the module name, unresolved arguments and the
values it provides.  These steps can be transformed into a workflow
specification or executed directly.

Once a set of candidate workflows has been produced via
``generate_workflows`` the synthesizer can serialise them for later reuse:

```python
workflows = synth.generate_workflows(start_module="module_a")
data = synth.to_dict()          # JSON serialisable mapping
path = synth.save()             # writes JSON/YAML to sandbox_data/generated_workflows

# create a .workflow.json file compatible with WorkflowDB
from workflow_synthesizer import save_workflow
save_workflow(workflows[0])
```
