# Energy Score Engine

`EnergyScoreEngine` is defined in `capital_management_bot.py` and computes a numeric score summarising the overall "energy" of the system. The score influences how aggressively other bots operate.

## Key features

- **Weight adaptation** – Each metric has a weight that updates when `reward` values are provided. The gradient step nudges weights towards features that correlate with higher rewards and normalises them afterwards.
- **Logistic scaling** – Raw metrics such as capital, profit trend and success rates are clamped and passed through a logistic function before being combined. This keeps the score in a stable `[0,1]` range.
- **Momentum smoothing** – Recent scores are blended using a momentum factor so that short‑term spikes do not cause wild oscillations.
- **External inputs** – Additional signals like `myelination`, `market` and `engagement` can be supplied to influence the score. The history of features and rewards is retained so an optional regression model can learn long‑term trends.
- **Prediction integration** – Prediction bots registered under the `"energy"` scope can adjust the computed score. `PredictionManager` assigns these bots and their forecasts are averaged with the base value.
- **Used by other bots** – `CapitalManagementBot.energy_score` exposes the engine to numerous helpers such as `ContrarianModelBot`, `ResourcePredictionBot`, `GAPredictionBot`, `GeneticAlgorithmBot`, `DatabaseManagementBot` and the `SelfImprovementEngine`.
- **ROI event logging** – `CapitalManagementBot.log_evolution_event` records ROI
  before and after each evolution cycle in a lightweight SQLite table so that
  future strategies can analyse the impact of changes.

The energy score provides a feedback loop that helps these bots scale up or conserve resources based on overall performance.
