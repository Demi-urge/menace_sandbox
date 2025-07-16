# Evolution Analysis Bot

`EvolutionAnalysisBot` trains a lightweight regression model on records stored in
`EvolutionHistoryDB`. The bot can predict the expected ROI of a particular
action (such as `self_improvement` or `system_evolution`) based on the ROI prior
to executing the action. These predictions let the orchestrator choose the most
promising strategy for the next cycle.

```python
from menace.evolution_analysis_bot import EvolutionAnalysisBot

analysis = EvolutionAnalysisBot()
analysis.train()
next_roi = analysis.predict("self_improvement", before_roi)
```
