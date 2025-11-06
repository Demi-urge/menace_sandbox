# Windows sandbox launcher testing guide

This guide outlines the manual verification checklist for the Windows sandbox launcher GUI, highlights the lightweight smoke tests that exercise the logging pipeline, and describes the environment prerequisites for both manual and automated coverage.

## Environment setup

Before running either the GUI or the automated checks, make sure the following requirements are satisfied:

- **Python 3.11+ with Tkinter** – the GUI instantiates native Tk widgets. Linux hosts may require the `python3-tk` package; Windows includes Tkinter by default.
- **Git CLI** – preflight uses `git fetch`/`git reset` to synchronise the repository. Ensure `git` is on `PATH` and can reach the remote origin.
- **Editable menace-sandbox install** – the preflight pipeline expects the repository to be available as an editable install. From the repository root run `python -m pip install -e .` to expose entry points.
- **Optional heavy dependencies** – the launcher will download model weights on demand. Pre-warming caches is recommended on constrained connections by executing `python -m neurosales.scripts.setup_heavy_deps --download-only` beforehand.
- **Test dependencies** – automated checks rely on `pytest` and `pytest-mock`. Install them with `python -m pip install -r requirements-dev.txt` or `python -m pip install pytest pytest-mock`.

### Known limitations

- Headless environments without an X server cannot display the GUI. The automated smoke tests patch Tk bindings and run without a display, but manual verification must occur on a host that can show Tk windows.
- The sandbox launch command spawns long-running subprocesses that cannot be fully exercised in unit tests. The manual checklist includes explicit operator confirmations for these flows.
- Preflight retries that rely on external network calls (Git, package index, registry priming) may still fail if the network is unreachable. Capture logs (`menace_gui_logs.txt`) and rerun once connectivity is restored.

## Manual test checklist

Run the launcher with `python -m tools.windows_sandbox_launcher_gui` on a machine that meets the setup prerequisites. Perform the following scenarios and record the outcome in the release checklist:

1. **Successful preflight path**
   - Ensure the repository is on the expected branch and dependencies are cached.
   - Click **Run Preflight** and confirm the status banner transitions through each step to "Preflight complete".
   - Verify the elapsed timer resets and the **Start Sandbox** button becomes enabled.

2. **Error pause dialog**
   - Induce a failure (for example temporarily remove network access before preflight).
   - Start preflight and wait for the error modal to appear.
   - Confirm the pause banner shows the failure summary, debug details capture the traceback, and the modal offers **Continue**, **Retry Step**, and **Abort** options.

3. **Retry logic**
   - After triggering the pause dialog, resolve the underlying issue (restore network access).
   - Choose **Retry Step** and ensure the failing action reruns without restarting earlier steps.
   - Confirm the status banner clears the error state and preflight resumes automatically.

4. **Negative decision handling**
   - During a pause, choose **Abort** from the dialog.
   - Verify the banner switches to an error state, the elapsed timer stops, and the **Run Preflight** button becomes available again while **Start Sandbox** remains disabled.

5. **Sandbox launch and shutdown**
   - After a successful preflight, click **Start Sandbox** and observe that the launch log entries stream into the GUI.
   - Once the sandbox reports ready, click **Stop Sandbox** and verify the process terminates and the GUI resets to the idle state.

Document any anomalies in the release notes and attach `menace_gui_logs.txt` excerpts for troubleshooting.

## Automated smoke tests

A pair of smoke tests covers the critical integrations that are difficult to exercise manually:

- **Log handler integration** – validates that `_initialise_file_logging` wires a queue handler, rotating file handler, and background listener without touching the filesystem when run under test doubles.
- **Log queue draining** – simulates queued GUI log entries and ensures `_drain_log_queue` flushes them into the Tk text widget, restores the disabled state, and reschedules polling.

Run the automated coverage with:

```bash
pytest unit_tests/test_windows_sandbox_gui_smoke.py
```

These checks provide fast feedback that the GUI can attach file logging and surface console output even when the visual components are mocked.
