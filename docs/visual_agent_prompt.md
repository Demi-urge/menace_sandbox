# Visual Agent Prompt Format

`BotDevelopmentBot` sends instructions to visual code generators. The prompt is
composed of several sections. Each section guides the agent to create a
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
6. **Metadata** – `_write_meta` saves the bot specification to a `meta.yaml`
   file in the repository root.
7. **Version control** – commit all generated files to git using descriptive
   messages.
8. **Testing** – run `scripts/setup_tests.sh` then execute `pytest --cov`,
   ensuring at least one test per function and reporting any failures.

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

### Metadata
`_write_meta` saves the specification to `meta.yaml` so future tools know how
the repository was generated.

### Version control
The agent should commit the repository to git using descriptive commit
messages once the files are created.

### Testing
Run `scripts/setup_tests.sh` and then execute `pytest --cov`. Ensure at least
one test per function. Example output:

```
================== test session starts ==================
tests/test_example_bot.py::test_basic PASSED [100%]
=================== 1 passed in 0.01s ===================
```

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
- `VISUAL_AGENT_TOKEN` – authentication token passed to the agent service.
- `BOT_DEV_HEADLESS` – set to `1` to disable interactive windows and run in
  headless mode.
- `VISUAL_AGENT_AUTOSTART` – set to `0` to prevent `run_autonomous` from
  launching a local visual agent when none is reachable.

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

### Testing
Run `pytest --cov`. Report any failures in the final message.
```
