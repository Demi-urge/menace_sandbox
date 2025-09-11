# Self Debugger Sandbox

`SelfDebuggerSandbox` analyses recent errors and tries generated patches in an isolated repository clone. Patch scores and flakiness history are stored in a local SQLite database by default. A `SelfCodingManager` must be provided to execute patches.

## Configuration

- `SANDBOX_SCORE_DB` – path of the local score history database.
- `PATCH_SCORE_BACKEND_URL` – optional remote backend used for storing and retrieving patch scores. Supports `http://` or `https://` endpoints and `s3://bucket/prefix` URLs.
- `WEIGHT_UPDATE_INTERVAL` – minimum seconds between score weight recalculations. Defaults to `60` and can also be supplied via the ``weight_update_interval`` constructor argument.

Essential environment variables:

- `SANDBOX_REPO_PATH` – path to the repository clone being patched.
- `SANDBOX_DATA_DIR` – directory where sandbox state such as ROI history is stored.

When a backend URL is configured the sandbox sends every patch score to the remote service and `recent_scores()` fetches records from the backend. If the backend is unreachable the sandbox falls back to the local database.

## Troubleshooting

- **Missing dependencies** – run `./setup_env.sh` to install required Python libraries.

## Synergy weights

`_composite_score` normalises recent metrics and feeds the result through a logistic
function. When the sandbox runs under `SelfImprovementEngine` the synergy metrics
(`synergy_roi`, `synergy_efficiency`, `synergy_resilience`, `synergy_antifragility`)
are multiplied by the engine's `synergy_learner.weights`. Custom mappings can be
supplied via the `weights` parameter of `_composite_score` to override the
learner values.
