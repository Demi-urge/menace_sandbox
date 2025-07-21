# Environment Generator

`environment_generator.generate_presets(count=None)` produces a list of dictionaries describing sandbox environment scenarios. Each dictionary contains resource limits and network settings that the sandbox runner reads from the environment before executing a code snippet.

The generator randomly chooses values for keys such as `CPU_LIMIT`, `MEMORY_LIMIT` and `DISK_LIMIT`. It can also introduce failure modes and emulate networking conditions.

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
- `memory` – restrict heap allocation.
- `timeout` – terminate the process prematurely.

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

`generate_presets` also accepts an optional `agent` and `tracker` argument:

```python
presets = generate_presets(3, agent=my_agent, tracker=my_tracker)
```

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
