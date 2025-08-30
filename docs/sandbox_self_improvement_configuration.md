# Sandbox and Self-Improvement Configuration Guide

This guide explains how to configure the sandbox together with self-improvement components.
It lists required dependencies, highlights security considerations, and offers troubleshooting
advice for common setup problems.

## Required dependencies
- Run `./setup_env.sh` or install packages from `requirements.txt` to provide core Python libraries.
- The self-improvement engine requires optional helpers. Missing modules such as
  `sandbox_runner` or `quick_fix_engine` raise a runtime error, so ensure they are
  available before enabling related features.
- Some sandbox features rely on external binaries like `ffmpeg` and `tesseract`.
  Verify they are on your `PATH` after installing the Python environment.

## Security considerations
- Resource and network limits are controlled through environment variables.
  `SECURITY_LEVEL` and `THREAT_INTENSITY` tune the simulated security posture,
  while `SANDBOX_REPO_PATH` and `SANDBOX_DATA_DIR` isolate repository and
  metrics storage locations.
- Keep tokens such as `VISUAL_AGENT_TOKEN` secret and run the sandbox under an
  unprivileged user to reduce risk. Apply OS-level isolation (containers or
  virtual machines) when evaluating untrusted code.

## Troubleshooting
### Missing modules
- Self-improvement cycles fail fast when helper modules are missing. Install
  `sandbox_runner` and `quick_fix_engine` or remove the feature flags that
  enable them.
- If Python dependencies are missing, rerun `./setup_env.sh` and check that
  `ffmpeg` and `tesseract` are installed.

### Cleanup errors
- Stray containers or QEMU overlays can accumulate if runs abort. Execute
  `python -m sandbox_runner.cli --cleanup` to remove leftovers, then
  `python -m sandbox_runner.cli check-resources` to confirm the cleanup.

### Planner availability
- `SandboxSettings` exposes an `enable_meta_planner` flag. When `True` the
  sandbox fails if `MetaWorkflowPlanner` is unavailable. Ensure the planner is
  installed or set the flag to `False` to proceed without meta-planning.

