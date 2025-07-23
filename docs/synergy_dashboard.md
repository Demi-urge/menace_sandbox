# Synergy Metrics Dashboard

`SelfImprovementEngine` exposes a small dashboard for reviewing synergy metrics recorded during sandbox runs. The dashboard expects `synergy_history.db` in the current directory by default and plots each metric over time. The sandbox now persists history in this SQLite file and will automatically migrate an older `synergy_history.json` on startup.

Start the server:

```bash
python -m menace.self_improvement_engine synergy-dashboard --port 8020
```

Use `--file` to override the location. The dashboard can run under different WSGI/ASGI servers:

```bash
python -m menace.self_improvement_engine synergy-dashboard --port 8020 --wsgi flask
python -m menace.self_improvement_engine synergy-dashboard --port 8020 --wsgi gunicorn
python -m menace.self_improvement_engine synergy-dashboard --port 8020 --wsgi uvicorn
```

Visit `http://localhost:8020/stats` for JSON averages and variance or `http://localhost:8020/plot.png` for a quick visualisation.

