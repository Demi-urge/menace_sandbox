# Environment Generator

`environment_generator.generate_presets(count=None, profiles=None)` produces a list of dictionaries describing sandbox environment scenarios. Each dictionary contains resource limits and network settings that the sandbox runner reads from the environment before executing a code snippet. When `profiles` is supplied it should be a list of named scenario profiles such as `"high_latency"` or `"hostile_input"`. The generator merges the parameters from these profiles with the randomly generated presets.

The generator randomly chooses values for keys such as `CPU_LIMIT`, `MEMORY_LIMIT` and `DISK_LIMIT`. It can also introduce failure modes and emulate networking conditions.

## Scenario Types

Presets fall into broad categories that target different behaviours:

- **baseline** – minimal limits used for control runs.
- **stress** – resource or network pressure such as `high_latency_api`.
- **security** – attack simulations like `hostile_input` or `user_misuse`.
- **concurrency** – thread and async task bursts via `concurrency_spike`.

Combine categories by listing multiple profiles when generating presets or on
the command line.

## Predefined Profiles

`generate_presets` ships with several canonical profiles that can be requested
via the `profiles` argument or by setting `SANDBOX_PRESET_MODE=canonical` in the
runner. Available names include:

- `high_latency_api` – adds heavy `NETWORK_LATENCY_MS` values.
- `hostile_input` – enables malicious stub generation.
- `user_misuse` – performs invalid API calls and file access attempts.
- `concurrency_spike` – defines `THREAD_BURST` and `ASYNC_TASK_BURST` bursts.

Generate a preset that combines hostile inputs with a concurrency spike:

```bash
python "$(python - <<'PY'
from dynamic_path_router import resolve_path
print(resolve_path('environment_cli.py'))
PY
)" --profiles hostile_input concurrency_spike --count 1
```

## Automatic Runner Coverage

`run_repo_section_simulations` ensures comprehensive testing by appending the
canonical profiles – `high_latency_api`, `hostile_input`, `user_misuse` and
`concurrency_spike` – whenever a custom preset list omits them. The runner also
requests additional module‑specific presets from the generator. These are built
through keyword matching on module paths so features like databases or parsers
receive scenarios tailored to their domain.

## Configuring and Extending Presets

Profiles are simple dictionaries merged into the randomly generated fields.
Add your own profile by defining it in a JSON file and passing its name to
`generate_presets`. Individual keys can be overridden via environment variables
or by editing the generated JSON. Example:

```bash
export CPU_LIMIT=2
python "$(python - <<'PY'
from dynamic_path_router import resolve_path
print(resolve_path('environment_cli.py'))
PY
)" --profiles hostile_input --count 1 --out presets.json
python "$(python - <<'PY'
from dynamic_path_router import resolve_path
print(resolve_path('run_autonomous.py'))
PY
)" --preset-file presets.json --runs 1
```

## Hostile Input Stub Strategy

The `hostile_input` profile or failure mode sets
`SANDBOX_STUB_STRATEGY=hostile` so that generated presets feed adversarial
payloads to each section. To enforce it without a profile:

```bash
export SANDBOX_STUB_STRATEGY=hostile
python "$(python - <<'PY'
from dynamic_path_router import resolve_path
print(resolve_path('sandbox_runner.py'))
PY
)" --runs 1
```

The `user_misuse` profile sets `SANDBOX_STUB_STRATEGY=misuse` which generates
stubs with missing fields or incorrect types to simulate user errors.

## Concurrency Settings

Profiles that include `concurrency_spike` populate the `THREAD_BURST` and
`ASYNC_TASK_BURST` fields to overwhelm concurrency controls. These values may
also be supplied manually:

```bash
export FAILURE_MODES=concurrency_spike
export THREAD_BURST=32
export ASYNC_TASK_BURST=128
python "$(python - <<'PY'
from dynamic_path_router import resolve_path
print(resolve_path('sandbox_runner.py'))
PY
)" --runs 1
```

## Available Parameters

- `CPU_LIMIT` – numeric string passed to `resource.setrlimit` to cap CPU seconds used by the snippet.
- `NETWORK_LATENCY_MS` – base latency in milliseconds applied via `tc netem` when network emulation is enabled.
- `SECURITY_LEVEL` – integer representing the strictness of the simulated security posture.
- `THREAT_INTENSITY` – percentage describing how aggressive the simulated adversary is.
- `GPU_LIMIT` – maximum number of GPUs exposed to the sandbox (passed through docker when supported).
- `OS_TYPE` – operating system flavour for container execution (e.g. `linux`, `windows`, `macos`).
- `CONTAINER_IMAGE` – explicit container image used when `OS_TYPE` targets Windows or macOS.
- `VM_SETTINGS` – dictionary describing the VM image and memory to use for non‑Linux tests. Set
  `windows_image` or `macos_image` to the path of your QEMU disk when `OS_TYPE`
  is `windows` or `macos`.

### Failure Modes

`generate_presets` may include a `FAILURE_MODES` key containing one or more of
the following values. When present the sandbox injects the corresponding fault
to stress error handling:

- `disk` – limit disk throughput or availability.
- `disk_corruption` – randomly corrupt files during writes.
- `network` – throttle networking and drop packets.
- `network_partition` – sever connectivity to selected hosts.
- `cpu` – reduce available CPU share.
- `cpu_spike` – introduce short bursts of high CPU usage.
- `concurrency_spike` – spawn bursts of threads and async tasks to stress concurrency handling. Presets include `THREAD_BURST` and `ASYNC_TASK_BURST` values.
- `memory` – restrict heap allocation.
- `timeout` – terminate the process prematurely.
- `hostile_input` – replace generated input stubs with malicious payloads. Internally this sets `SANDBOX_STUB_STRATEGY` to `hostile`.
- `user_misuse` – supply wrong types or omit fields, setting `SANDBOX_STUB_STRATEGY=misuse` and logging the resulting errors.

Include the key as a list or comma separated string. The sandbox runner reads
`FAILURE_MODES` from the environment preset before executing each snippet and
activates the listed behaviours.

Other generated values include memory, disk and GPU limits, bandwidth settings and packet loss rates. When these variables are present the sandbox adjusts resource limits, available GPUs and network behaviour accordingly to mimic the desired environment. If `OS_TYPE` is specified the sandbox attempts to use a matching container image when executing code. Provide your own QEMU images under `VM_SETTINGS` when targeting Windows or macOS so the sandbox can boot the correct guest.

Use the function like so:

```python
from menace.environment_generator import generate_presets

presets = generate_presets(3)
for preset in presets:
    print(preset)
```

These presets can be passed directly to `run_repo_section_simulations` or `_run_sandbox` for scenario testing.

Example preset with hostile input stubs (`SANDBOX_STUB_STRATEGY` set to `hostile`):

```json
{
  "CPU_LIMIT": "1",
  "MEMORY_LIMIT": "512Mi",
  "FAILURE_MODES": "hostile_input",
  "SANDBOX_STUB_STRATEGY": "hostile"
}
```

`generate_presets` also accepts an optional `agent` and `tracker` argument:

```python
presets = generate_presets(3, agent=my_agent, tracker=my_tracker)
```

Profiles can be selected explicitly:

```python
presets = generate_presets(profiles=["high_latency", "hostile_input"])
```

## Interpreting Per-Scenario Metrics

Each preset carries a `SCENARIO_NAME` so that ROI and synergy measurements are
grouped per scenario. After a run inspect `roi_history.json` or the metrics
database and filter by this name to see how a particular profile behaves. A drop
in `synergy_resilience` for `hostile_input` indicates the system struggled with
malicious payloads, while high `security_score` under `high_latency_api` shows
robust network handling.

When provided and the tracker contains at least five ROI samples, the agent's
`decide()` method adjusts `CPU_LIMIT`, `MEMORY_LIMIT`, `BANDWIDTH_LIMIT` and
`THREAT_INTENSITY` for each preset. The agent's state is saved so future runs
continue refining the policy based on previous ROI outcomes.

## Adaptive Presets

`environment_generator.adapt_presets(tracker, presets)` adjusts an existing list
of presets based on ROITracker history. When the recent average `security_score`
is high the function increases `THREAT_INTENSITY` for further exploration. Low
scores result in less aggressive presets.

Additional synergy metrics influence the adjustments:

- **synergy_efficiency** – positive values lower `CPU_LIMIT` and `MEMORY_LIMIT`
  because modules perform better together. Negative values scale the limits up.
- **synergy_adaptability** – similar to efficiency, higher values reduce `CPU_LIMIT` and `MEMORY_LIMIT` while negative values increase them.
- **synergy_antifragility** – when above the threshold the generator increases
  `THREAT_INTENSITY` to test how well the system benefits from extra stress;
  negative trends decrease the intensity.
- **synergy_resilience** – controls bandwidth limits. Higher values increase
  `BANDWIDTH_LIMIT`, `MIN_BANDWIDTH` and `MAX_BANDWIDTH` while negative values
  reduce them.
- **synergy_safety_rating** – adjusts `THREAT_INTENSITY` based on combined
  safety performance; positive values increase the intensity while negative
  values decrease it.
- **synergy_reliability** – higher values tighten adjustments based on more
  trustworthy synergy forecasts.
- **synergy_maintainability** – influences CPU limits to encourage easier upkeep.
- **synergy_throughput** – scales bandwidth when modules sustain higher output.

Synergy *predictions* are also consulted when present. `ROITracker.predict_synergy()`
forecasts the next `synergy_roi` value and `predict_synergy_metric()` provides
predictions for metrics like `efficiency` or `resilience`. These forecasts are
averaged with the most recent measurements so `adapt_presets` can pre‑emptively
raise or lower CPU, memory and network limits before each sandbox iteration.

When the environment variable `SANDBOX_PRESET_RL_PATH` points to a writable
file, `adapt_presets` uses :class:`PresetRLAgent` to learn from the recorded ROI
and synergy values. Once at least three ROI samples exist the agent predicts
actions for CPU, memory, bandwidth and threat intensity. The learned policy is
stored at the provided path so adjustments improve over time. If the variable is
unset the policy is stored in ``sandbox_data/preset_policy.json`` and reloaded
automatically on the next run. If history is too short the function falls back
to the heuristics described above.

Set ``SANDBOX_PRESET_RL_STRATEGY`` to choose the reinforcement learning
algorithm. When unset the agent defaults to ``q_learning``.

If ``SANDBOX_ADAPTIVE_AGENT_PATH`` is set, ``adapt_presets`` employs
``AdaptivePresetAgent`` to refine presets using a reinforcement learning
policy. The algorithm can be selected via ``SANDBOX_ADAPTIVE_AGENT_STRATEGY``
and defaults to ``q_learning``. Set this variable to ``double_dqn`` to enable
the more sophisticated Double DQN strategy (PyTorch is used when available).
The agent stores its learned policy at the configured path and reloads it on
startup.
## Preset Policy CLI

Use `preset_policy_cli.py` to persist or restore the reinforcement learning policy used by `adapt_presets`.

```bash
python preset_policy_cli.py export --out policy.json
python preset_policy_cli.py import policy.json
```

The tool calls `export_preset_policy` and `import_preset_policy` under the hood, encoding the policy table as JSON.

## Example: RL adaptation with run_autonomous.py

Follow these steps to launch the autonomous sandbox with reinforcement
learning enabled for preset adaptation:

1. Set the path where the learned policy should be stored:

   ```bash
   export SANDBOX_PRESET_RL_PATH=sandbox_data/preset_policy.json
   ```

   The `PresetRLAgent` persists its state at this location so future runs can
   resume learning.

2. (Optional) Choose the RL algorithm by setting
   `SANDBOX_PRESET_RL_STRATEGY`:

   ```bash
   export SANDBOX_PRESET_RL_STRATEGY=double_dqn
   ```

   When unset the agent defaults to `q_learning`.

3. Run the sandbox using `run_autonomous.py`:

   ```bash
   python "$(python - <<'PY'
from dynamic_path_router import resolve_path
print(resolve_path('run_autonomous.py'))
PY
)" --preset-count 2 --runs 1
   ```

   During each cycle `environment_generator.adapt_presets` loads the policy
   from `SANDBOX_PRESET_RL_PATH`, adjusts CPU, memory, bandwidth and threat
   levels based on recent ROI and synergy values, then writes the updated
   policy back to disk.

Synergy metrics such as `synergy_efficiency`, `synergy_resilience`,
`synergy_antifragility`, `synergy_reliability`, `synergy_maintainability` and
`synergy_throughput` indicate how well modules cooperate. Positive scores
tighten resource limits while negative ones relax them. The RL agent also
consumes `synergy_roi` alongside ROI history so improvements in these metrics
directly influence the policy update step.

## LLM stub generation

Set `SANDBOX_STUB_MODEL` to a model name understood by the configured backend
to enable language-model driven stub creation. Generated objects are cached
alongside locally produced stubs under `SANDBOX_STUB_CACHE`. When no model is
configured the sandbox falls back to a deterministic rule-based strategy.
