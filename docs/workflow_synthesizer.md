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
