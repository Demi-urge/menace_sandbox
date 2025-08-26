# Workflow Evolution Bot

`WorkflowEvolutionBot` analyses usage data recorded in `PathwayDB` and suggests improved workflow sequences. The `analyse()` method looks up the most frequent pathways, gathers their average ROI and returns a list of `WorkflowSuggestion` objects containing the sequence and expected ROI.

The bot can also generate alternative sequences through step swaps and intent-based module injection. `generate_variants()` yields these candidate sequences which can be fed into the orchestrator or experiment manager.

```python
from menace.workflow_evolution_bot import WorkflowEvolutionBot

bot = WorkflowEvolutionBot()
for seq in bot.generate_variants(limit=3):
    print(seq)
```

`BotCreationBot` can use this bot to prioritise new plans. When passed via the
`workflow_bot` parameter, suggestions returned by `analyse()` are sorted to the
front of the candidate list before any trending keywords are applied.
