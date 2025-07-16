# Evolution Pipeline Flow

The diagram below summarizes how Menace adapts itself when metrics degrade.

```mermaid
flowchart TD
    A[Metrics drop detected] --> B[SelfImprovementEngine]
    B --> C[SelfCodingManager]
    C --> D[EvolutionOrchestrator]
    D -->|ROI stable| F[Next cycle]
    D -->|ROI down| E[SystemEvolutionManager]
    E --> G[WorkflowEvolutionBot]
    G --> H[Deployment update]
    H --> F
```

The process begins when `DataBot` notices an error spike or low energy score.
`EvolutionOrchestrator` chooses between running a self‑improvement cycle or a
larger structural evolution. Predictions from `EvolutionPredictor` guide the
choice. When structural changes succeed the updated bots are redeployed and the
loop starts again.
