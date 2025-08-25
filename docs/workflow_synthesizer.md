# Workflow Synthesizer

`WorkflowSynthesizer` expands a seed module by blending structural signals from
`ModuleSynergyGrapher` with semantic intent search.  It inspects module inputs
and outputs to resolve dependency order and emits small workflow candidates.

## Quick start

```python
from workflow_synthesizer import WorkflowSynthesizer
from workflow_spec import to_spec, save

synth = WorkflowSynthesizer()
steps = synth.generate_workflows("module_a", problem="summarise data")[0]

spec = to_spec([
    {"name": s["module"], "bot": s["module"], "args": s["inputs"]}
    for s in steps
])
save(spec)  # writes module_a.workflow.json and registers it with WorkflowDB
```

Workflows serialised with :func:`workflow_spec.save` are compatible with
``WorkflowDB`` and can be ingested directly.
