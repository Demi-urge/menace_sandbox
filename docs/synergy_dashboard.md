# Synergy Metrics Dashboard

`SelfImprovementEngine` exposes a small dashboard for reviewing synergy metrics recorded during sandbox runs. The dashboard reads `synergy_history.json` and plots each metric over time.

Start the server:

```bash
python -m menace.self_improvement_engine synergy-dashboard --file sandbox_data/synergy_history.json --port 8020
```

Visit `http://localhost:8020/stats` for JSON averages and variance or `http://localhost:8020/plot.png` for a quick visualisation.

