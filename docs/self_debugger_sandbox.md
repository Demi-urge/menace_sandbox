# Self Debugger Sandbox

`SelfDebuggerSandbox` analyses recent errors and tries generated patches in an isolated repository clone. Patch scores and flakiness history are stored in a local SQLite database by default.

## Configuration

- `SANDBOX_SCORE_DB` – path of the local score history database.
- `PATCH_SCORE_BACKEND_URL` – optional remote backend used for storing and retrieving patch scores. Supports `http://` or `https://` endpoints and `s3://bucket/prefix` URLs.

When a backend URL is configured the sandbox sends every patch score to the remote service and `recent_scores()` fetches records from the backend. If the backend is unreachable the sandbox falls back to the local database.
