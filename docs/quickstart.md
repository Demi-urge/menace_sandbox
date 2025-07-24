# Quickstart

The Menace sandbox can be launched directly from the repository once dependencies are installed.

```bash
python sandbox_runner.py full-autonomous-run --preset-count 3
```

This starts an iterative self-improvement loop using randomly generated environment presets. The script copies the repository to a temporary directory so the original files and databases remain untouched. Metrics are stored in `sandbox_data/roi_history.json` and can be visualised using the metrics dashboard.

GPT powered features such as brainstorming and prompt generation require the `OPENAI_API_KEY` environment variable. Create a `.env` file with this key or export it in your shell before running the command.

Run `./setup_env.sh` to install necessary packages and create a basic environment file via `auto_env_setup.ensure_env()` if one doesn't exist.

To automatically start the local visual agent and run the sandbox in one step use:
```bash
python scripts/run_personal_sandbox.py
```
