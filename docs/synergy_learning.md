# Synergy learning

This page explains how synergy weights are learned and how predictions feed into ROI calculations.

## SynergyWeightLearner

`SynergyWeightLearner` maintains seven weights controlling how synergy metrics influence policy updates:

``roi``
``efficiency``
``resilience``
``antifragility``
``reliability``
``maintainability``
``throughput``

The learner stores these values in `synergy_weights.json`. Each cycle `SelfImprovementEngine._update_synergy_weights` computes the latest deltas for `synergy_<name>` metrics and calls `SynergyWeightLearner.update`. The default implementation uses an actor–critic policy to nudge weights toward positive ROI changes. A `DQNSynergyLearner` subclass provides a deeper Double DQN variant when PyTorch is available.

Weights can be edited manually or with `synergy_weight_cli.py`. Environment variables such as `SYNERGY_WEIGHT_ROI` or `SYNERGY_WEIGHT_EFFICIENCY` override the loaded values at startup. The learning rate is controlled by `SYNERGY_WEIGHTS_LR`.

## ARIMASynergyPredictor

`ARIMASynergyPredictor` fits an ARIMA model to a synergy metric series and predicts the next value. `ROITracker.predict_synergy` and `predict_synergy_metric` consult this predictor when the environment variable `SANDBOX_SYNERGY_MODEL=arima` is set and enough history is available. Otherwise a simpler exponential moving average is used.

## ROI feedback loop

During ROI calculation `SelfImprovementEngine` adds the weighted `synergy_roi` delta to the profit figure and adjusts the energy score using `synergy_efficiency`, `synergy_resilience` and `synergy_antifragility`. This means synergy metrics directly impact ROI as the weights change. Improving prediction accuracy through models such as `ARIMASynergyPredictor` therefore helps the engine react earlier to beneficial or harmful interactions.

## Tuning examples

Personal deployments can start with custom weights and learning rates:

```dotenv
# .env
SYNERGY_WEIGHT_ROI=1.2
SYNERGY_WEIGHT_EFFICIENCY=0.8
SYNERGY_WEIGHT_RESILIENCE=1.0
SYNERGY_WEIGHTS_LR=0.05
```

Create a minimal `synergy_weights.json`:

```json
{"roi": 1.2, "efficiency": 0.8, "resilience": 1.0,
 "antifragility": 1.0, "reliability": 1.0,
 "maintainability": 1.0, "throughput": 1.0}
```

Then run:

```bash
python synergy_weight_cli.py --path synergy_weights.json show
```

After a few sessions record synergy history and train the weights:

```bash
python synergy_weight_cli.py --path synergy_weights.json train sandbox_data/synergy_history.json
```

The updated file persists between runs and influences ROI calculations automatically.
