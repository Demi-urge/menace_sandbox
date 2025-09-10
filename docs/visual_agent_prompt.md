# Visual Agent Prompt Format

`BotDevelopmentBot` sends instructions to visual code generators through the
local ``SelfCodingEngine``. The prompt is composed of several sections. Each
section guides the agent to create a
complete, testable repository. The list below summarises the required headings:

1. **Introduction** – states the target language, bot name and short
   description.
2. **Functions** – bullet list of required functions extracted from the
   handoff specification.
3. **Dependencies** – packages that must be available via
   `requirements.txt`.
4. **Coding standards** – PEP8 compliance with 4-space indents and lines under
   79 characters, Google style docstrings and inline comments for complex logic.
5. **Repository layout** – the main module is `<bot_name>.py`, a
   `requirements.txt` lists dependencies, tests live under `tests/` with
   `test_*.py` filenames and a `README.md` summarises usage.
6. **Environment** – includes the Python version used by the sandbox.
7. **Metadata** – `_write_meta` saves the bot specification to a `meta.yaml`
   file in the repository root.
8. **Version control** – commit all changes to git using descriptive commit
   messages.
9. **Testing** – run `scripts/setup_tests.sh` then execute `pytest --cov`,
   ensuring at least one test per function and reporting any failures.
10. **Snippet context** – relevant code lines added by the sandbox.

This structure helps the agent produce consistent, testable starter
repositories. The sections must appear in the order listed above so that the
visual agent can parse them reliably.

## Prompt sections in detail

### Introduction
Provides context for the code generation request. It typically looks like:

```
### Introduction
Generate Python code for `ExampleBot`. This bot processes event logs.
```

### Functions
Each function is listed in signature form. Example:

```
### Functions
- `load_events(path: str) -> list[dict]`
- `filter_events(events: list[dict]) -> list[dict]`
```

### Dependencies
All packages required to run the bot are declared here:

```
### Dependencies
requests>=2.31
```

### Coding standards
Follow PEP8 with 4-space indents and lines under 79 characters. Provide Google
style docstrings for modules, classes and functions. Add inline comments to
explain complex logic.

### Repository layout
Lists the files that must exist. The main module is named after the bot and is
placed at the repository root. Dependencies belong in `requirements.txt`.
Tests live in `tests/` with filenames matching `test_*.py`. A `README.md`
describes usage and setup steps.

### Environment
Lists the Python version running in the sandbox so the agent can tailor
generated code accordingly. Example:

```
### Environment
3.11.12
```

### Metadata
`_write_meta` saves the specification to `meta.yaml` so future tools know how
the repository was generated.

### Version control
The agent should commit all changes to git using descriptive commit messages
once the files are created.

### Testing
Run `scripts/setup_tests.sh` and then execute `pytest --cov`. Ensure at least
one test per function. Example output:

```
================== test session starts ==================
tests/test_example_bot.py::test_basic PASSED [100%]
=================== 1 passed in 0.01s ===================
```

### Snippet context
`SelfCodingEngine` appends snippet lines here. Each snippet starts with the file
path and line numbers followed by the code itself so the agent can reuse or
modify existing logic.

## Example repository structure

```
my_bot/
├── my_bot.py
├── requirements.txt
├── README.md
└── tests/
    └── test_my_bot.py
```

## Running tests

Before executing tests the visual agent must run `scripts/setup_tests.sh` to
install packages listed in `requirements.txt`. After that execute
`pytest --cov` inside the repository. Any failing tests must be reported in
full so they can be fixed before deployment.

## Environment variables

`BotDevConfig` exposes several environment variables that influence how the
visual agent is contacted. The most relevant are:

- `VISUAL_AGENT_URLS` – semicolon separated list of agent endpoints. If unset
  the values of `VISUAL_DESKTOP_URL` and `VISUAL_LAPTOP_URL` are used.
- `VISUAL_AGENT_TOKEN` – authentication token passed to the agent service. The
  variable is mandatory and the server exits when it is missing.
- `MENACE_AGENT_PORT` – port where `menace_visual_agent_2.py` listens for HTTP connections.
- `BOT_DEV_HEADLESS` – set to `1` to disable interactive windows and run in
  headless mode.
 - `VISUAL_AGENT_AUTOSTART` – set to `0` to prevent `run_autonomous` from
  launching a local visual agent when none is reachable.
 - `VISUAL_AGENT_AUTO_RECOVER` – `1` by default; set to `0` to disable automatic queue recovery.
- `VA_PROMPT_TEMPLATE` – path to a template (or inline template string) used to
  build the visual agent prompt. The template receives `{path}`,
  `{description}`, `{context}` and `{func}` placeholders.
- `VA_PROMPT_PREFIX` – additional text prepended before the generated prompt.
- `VA_REPO_LAYOUT_LINES` – number of repository layout lines to include.
- `VA_MESSAGE_PREFIX` – overrides the hard-coded prefix used by
  `VisualAgentClient` when forwarding messages. By default, the client
  prepends "Improve Menace by enhancing error handling and modifying
  existing bots." (see `DEFAULT_MESSAGE_PREFIX` in
  `visual_agent_client.py`).
  - `VISUAL_AGENT_STATUS_INTERVAL` – poll `/status` periodically to record queue
    depth for the dashboard. Set to `0` to disable.

When `run_autonomous.py` is used a `VisualAgentMonitor` thread keeps the
service running. It polls the agent's `/health` endpoint and restarts
`menace_visual_agent_2.py` with `/recover` if the service stops responding.
The agent itself runs a queue watchdog that rebuilds the database and restarts
the worker when corruption or crashes are detected. Manual commands such as
`--recover-queue` or `--repair-running` are therefore typically unnecessary.

## Visual agent service

The standalone service `menace_visual_agent_2.py` exposes HTTP endpoints on the
port specified by `MENACE_AGENT_PORT` (default 8001). Requests must include
`VISUAL_AGENT_TOKEN` for authentication. The `/run` endpoint always returns
HTTP `202` together with a task id. Submitted jobs are appended to
`visual_agent_queue.db` and executed sequentially. Poll `/status/<id>` to check
progress. The queue database is created automatically inside
`SANDBOX_DATA_DIR` and crash recovery and stale lock handling are automatic as
detailed above. The queue uses SQLite for persistence and a
`visual_agent_state.json` file records job status and the timestamp of the last
completed task.

The `/status` endpoint returns a JSON object with two fields:

- `active` – `true` while a job is running and `false` when idle.
- `queue` – number of queued jobs waiting for execution.

Jobs are processed sequentially so at most one task runs at a time regardless of how many clients enqueue work.
If the service fails to start due to a stale lock run `python menace_visual_agent_2.py --cleanup` then repair the queue with `--repair-running --recover-queue`. These commands are mainly for troubleshooting because `run_autonomous.py` automatically restarts the service and triggers recovery. Use `--resume` to process tasks without starting the server.

## Prompt customisation


`VA_MESSAGE_PREFIX` overrides the instructions prepended before each message.
`VA_PROMPT_PREFIX` adds extra text to the start of the prompt while
`VA_PROMPT_TEMPLATE` replaces the default layout entirely. The template receives
`{path}`, `{description}`, `{context}` and `{func}` so you can insert snippet
metadata wherever desired.

A minimal template might look like:

```text
### Introduction
Add {func} to {path}. It should {description}.

### Snippet context
{context}
```

Save this file as `va.tmpl` and pass it to the sandbox with
`VA_PROMPT_TEMPLATE=va.tmpl`:

```bash
VA_PROMPT_PREFIX="[internal]" VA_PROMPT_TEMPLATE=va.tmpl python "$(python - <<'PY'
from dynamic_path_router import resolve_path
print(resolve_path('sandbox_runner.py'))
PY
)" full-autonomous-run
```

## Sample prompt

```
### Introduction
Generate Python code for `ImageFilterBot`. This bot filters low-quality images from a directory.

### Functions
- `filter_images(path: str) -> list[str]`
- `score_image(path: str) -> float`

### Dependencies
pillow==9.0.0

### Coding standards
Follow PEP8 with 4-space indents and <79 character lines. Include Google-style docstrings for modules, classes and functions. Add inline comments explaining complex logic.

### Repository layout
image_filter_bot.py
requirements.txt
README.md
tests/test_image_filter_bot.py

### Environment
3.11.12

### Metadata
description: filter low-quality images from a directory

### Version control
commit all changes to git using descriptive commit messages

### Testing
Run `scripts/setup_tests.sh` then execute `pytest --cov`. Report any failures in the final message.

### Snippet context
image_filter_bot.py:10-12
def helper():
    pass
```
