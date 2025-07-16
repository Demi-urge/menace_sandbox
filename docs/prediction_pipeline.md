# Prediction Pipeline

Menace uses a small ecosystem of bots to train and evolve forecasting models. `DataBot` records metrics for each run into a `MetricsDB` while `CapitalManagementBot` computes profit-based scores from that data. Both `GAPredictionBot` and `GeneticAlgorithmBot` query these metrics when they evolve new models and also report their own metrics back through `DataBot`.

## Bot Evaluation

`PredictionManager.monitor_bot_performance` reviews metrics and ROI for every registered prediction bot. When the measured accuracy or overall score falls below a threshold the bot is retired and `PredictionManager.trigger_evolution` creates a replacement.

## Training Pipeline

`PredictionTrainingPipeline.train` takes one or more bot IDs and keeps evolving them until their computed accuracy exceeds a configured threshold or the maximum generation limit is hit. It relies on `trigger_evolution` to spawn new bots and logs each cycle via `DataBot.collect` and `CapitalManagementBot.update_rois`.

The pipeline also exposes a simple ROI forecasting helper using recent metrics. `PredictionTrainingPipeline._roi_forecast()` fits a linear regression over the last few ROI values and records the predicted next ROI via `MetricsDB.log_eval`.

## Data Flow

1. **DataBot** captures `cpu`, `memory`, `response_time`, `errors`, `revenue` and `expense` for each bot.
2. **CapitalManagementBot** reads these metrics to calculate ROI and energy scores. The current average energy score is logged alongside each metric collection.
3. **DataBot** also stores the patch success rate from `PatchHistoryDB` so other bots can see how well code updates perform.
4. **GeneticAlgorithmBot** and **GAPredictionBot** evolve models using those scores and store new metrics through `DataBot`.
5. **PredictionManager** calls `trigger_evolution` whenever no matching bot exists or monitoring detects degradation, ensuring a fresh bot is registered.

### Usage Hints

```python
manager = PredictionManager(data_bot=data_bot, capital_bot=capital)
manager.monitor_bot_performance(data_bot.db)
results = PredictionTrainingPipeline(manager, data_bot, capital).train([bot_id])
```

These utilities keep the prediction bots healthy and automatically evolve them when performance drops.

