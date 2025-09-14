# Sandbox and Self-Improvement Configuration Guide

This guide explains how to configure the sandbox together with self-improvement components.
It lists required dependencies, highlights security considerations, and offers troubleshooting
advice for common setup problems.

## Required dependencies
- Run `./setup_env.sh` or install packages from `requirements.txt` to provide core Python libraries.
- The self-improvement engine requires optional helpers. Missing modules such as
  `sandbox_runner.orphan_integration` or `quick_fix_engine` raise a runtime
  error, so ensure they are available before enabling related features.
- Some sandbox features rely on external binaries like `ffmpeg` and `tesseract`.
  Verify they are on your `PATH` after installing the Python environment.

## Policy hyperparameters
- `POLICY_ALPHA` – learning rate applied during policy updates.
- `POLICY_GAMMA` – discount factor for future rewards.
- `POLICY_EPSILON` – exploration rate for epsilon-greedy strategies.
- `POLICY_TEMPERATURE` – temperature for softmax exploration.
- `POLICY_EXPLORATION` – exploration strategy to use, such as `epsilon_greedy`.

## Snapshot and regression settings
- `ENABLE_SNAPSHOT_TRACKER` – toggles capturing metric snapshots and computing
  deltas between improvement cycles.
- `ROI_DROP_THRESHOLD` – ROI delta at or below this value marks a regression
  and records a failed prompt attempt.
- `ENTROPY_REGRESSION_THRESHOLD` – drop in entropy at or below this threshold
  is treated as a regression during patch evaluation.

## Self-coding thresholds
Per-bot regression limits are stored in `config/self_coding_thresholds.yaml`.
Each entry defines the maximum allowed ROI drop, error increase and test
failure increase before a bot is considered degraded. A custom test command can
also be provided. Example:

```yaml
default:
  roi_drop: -0.1
  error_increase: 1.0
  test_failure_increase: 0.0
  test_command: ["pytest", "-q"]
bots:
  QuickFixEngine:
    roi_drop: -0.1
    error_increase: 1.0
    test_failure_increase: 0.0
    test_command: ["pytest", "tests/quickfix", "-q"]
  BotDevelopmentBot:
    roi_drop: -0.1
    error_increase: 1.0
    test_failure_increase: 0.0
  ImplementationOptimiserBot:
    roi_drop: -0.1
    error_increase: 1.0
    test_failure_increase: 0.0
```

The default test command is `pytest -q`.  Override it globally with the
`SELF_CODING_TEST_COMMAND` environment variable or the corresponding
`SandboxSettings.self_coding_test_command` field.  When
``PatchApprovalPolicy`` is constructed without an explicit ``test_command``
it queries the active ``threshold_service`` and falls back to
``SandboxSettings.bot_thresholds`` for per-bot overrides.  The selected
command can be changed on a running policy via
``PatchApprovalPolicy.update_test_command``.

## Security considerations
- Resource and network limits are controlled through environment variables.
  `SECURITY_LEVEL` and `THREAT_INTENSITY` tune the simulated security posture,
  while `SANDBOX_REPO_PATH` and `SANDBOX_DATA_DIR` isolate repository and
  metrics storage locations.
- Keep API tokens secret and run the sandbox under an unprivileged user to
  reduce risk. Apply OS-level isolation (containers or virtual machines) when
  evaluating untrusted code.

## Troubleshooting
### Missing modules
- Self-improvement cycles fail fast when helper modules are missing. Install
  `sandbox_runner` (with the `sandbox_runner.orphan_integration` module) and
  `quick_fix_engine` or remove the feature flags that enable them.
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

