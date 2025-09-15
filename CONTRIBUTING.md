# Contributor Guidelines

## Path resolution

All new code should locate files through `dynamic_path_router`. Use
`resolve_path` or `path_for_prompt` instead of hard-coded relative paths so the
project remains portable across different checkouts.

## SQLite connections

Use `DBRouter` for all database access. Direct calls to `sqlite3.connect` are not
permitted outside approved scaffolding scripts
(`scripts/new_db.py`, `scripts/new_db_template.py`, `scripts/scaffold_db.py`, and
`scripts/new_vector_module.py`).  In the rare case a raw connection is required,
include explicit audit logging and add the module to
`tests/approved_sqlite3_usage.txt` with a justification.

To avoid accidental direct connections, CI runs
`tests/test_db_router_enforcement.py` and a pre-commit hook which executes
`scripts/check_sqlite_connections.py` against all Python sources. These checks
fail the build if an unapproved `sqlite3.connect` call is detected. Legitimate
new uses must be added to `tests/approved_sqlite3_usage.txt` together with an
explanation for the exemption.

## Dynamic path resolution

Hard coding repository-relative paths (e.g., strings containing `sandbox_data`
or path fragments like `foo/bar.py`) can break when the project moves. Use
`dynamic_path_router.resolve_path` to build such paths dynamically. The resolver
respects environment overrides like `MENACE_ROOT` or `SANDBOX_REPO_PATH`,
keeping scripts portable. The pre-commit hook
`tools/check_dynamic_paths.py` enforces this rule by failing if a file contains
these patterns without a corresponding `resolve_path` call.

## Static path references

Avoid hard coding string literals that end with `.py`. Such paths must be
wrapped by `resolve_path` or `path_for_prompt` to remain portable across forks
and clones. The `tools/check_static_paths.py` pre-commit hook scans for these
violations.
CI enforces this rule by running `python tools/check_static_paths.py $(git ls-files '*.py')`,
and the workflow fails if any static path is detected.
Run this command locally (or `pre-commit run check-static-paths --all-files`)
before pushing changes to catch issues early.

## `dynamic_path_router.resolve_path`

`dynamic_path_router.resolve_path` returns an absolute path within the
repository. Use it whenever code, scripts, or tests need to locate files so the
project works from any checkout. The resolver's behaviour and advanced options
are documented in [docs/dynamic_path_router.md](docs/dynamic_path_router.md).

**Scripts**

```bash
python "$(python - <<'PY'
from dynamic_path_router import resolve_path
print(resolve_path('sandbox_runner.py'))
PY
)" --help
```

**Tests**

```python
from dynamic_path_router import resolve_path

def test_example():
    data = resolve_path('sandbox_data/example.json')
    assert data.exists()
```

Run `pre-commit run check-static-paths --all-files` before committing to ensure
no hard-coded `.py` paths slip through.

## ContextBuilder dependency

Every constructor or function that assembles prompts must accept a
`ContextBuilder` and use it for token accounting and ROI tracking. This applies
to helpers like `_build_prompt`, `build_prompt`, `generate_patch`, and
`generate_candidates`. Instantiate the builder with local database paths at the
call site and pass it explicitly; components should fail fast when the argument
is missing and must not define a default for the `context_builder` parameter.
In rare cases where a default is unavoidable or a call intentionally omits the
builder, append `# nocb` to the line to document the exception.

```python
from vector_service.context_builder import ContextBuilder
from prompt_engine import PromptEngine
from bot_development_bot import BotDevelopmentBot, BotSpec
from automated_reviewer import AutomatedReviewer
from quick_fix_engine import QuickFixEngine, generate_patch
from error_bot import ErrorDB
from self_coding_engine import SelfCodingEngine
from model_automation_pipeline import ModelAutomationPipeline
from data_bot import DataBot
from bot_registry import BotRegistry
from self_coding_manager import SelfCodingManager

builder = ContextBuilder("bots.db", "code.db", "errors.db", "workflows.db")

engine = PromptEngine(context_builder=builder)
prompt = engine.build_prompt("Expand tests", context_builder=builder)

dev_bot = BotDevelopmentBot(context_builder=builder)
dev_bot._build_prompt(BotSpec(name="demo", purpose="Add feature"), context_builder=builder)

reviewer = AutomatedReviewer(context_builder=builder)

manager = SelfCodingManager(
    SelfCodingEngine(),
    ModelAutomationPipeline(),
    data_bot=DataBot(),
    bot_registry=BotRegistry(),
)
qfe = QuickFixEngine(ErrorDB(), manager, context_builder=builder)
generate_patch("sandbox_runner", context_builder=builder)
```

The pre-commit hook `check-context-builder-usage` wraps
`scripts/check_context_builder_usage.py` and is required for submissions. Run:

```bash
pre-commit run check-context-builder-usage --all-files
```

Continuous integration runs this hook and fails the build when missing or
implicit `ContextBuilder` usage is detected. The script also flags disallowed
defaults for `context_builder` parameters (including sentinel objects), calls to
helpers like `generate_candidates` that omit a `context_builder` keyword, or
imports of `get_default_context_builder`, and CI fails when these patterns
appear.  Literal prompt strings passed directly to LLM clients are likewise
rejected unless produced by `ContextBuilder.build_prompt` or
`SelfCodingEngine.build_enriched_prompt`.  Direct `Prompt(...)` calls are banned
outside `vector_service/context_builder.py`, and direct calls to
`PromptEngine.build_prompt` are disallowed.

When `pre-commit run check-context-builder-usage --all-files` reports
"Prompt instantiation disallowed", replace the `Prompt(...)` constructor with
`context_builder.build_prompt(...)` or
`SelfCodingEngine.build_enriched_prompt(...)` and pass the resulting dataclass
directly to the client. Only modules listed in the hook's whitelist may build
`Prompt` objects directly. In exceptional cases the warning can be suppressed
with `# nocb`, but refactoring to use the standard builders is preferred.

## Coding bot registration

All new coding bots must be created via the `@self_coding_managed` decorator
from `coding_bot_interface`. Pre-commit and CI run
`tools/find_unmanaged_bots.py`, which exits with a non-zero status when any
`*_bot.py` lacks this decorator, causing the build to fail. Decorating a bot
automatically registers it with `BotRegistry` and wires ROI and error metrics
into `DataBot`, allowing the system to track and improve the bot over time. Bot
constructors **must** accept `bot_registry`, `data_bot`, and
`selfcoding_manager` parameters and forward them to the decorator to ensure
proper registration. Avoid instantiating new coding bots without this decorator.
When a module needs a `SelfCodingManager`, invoke
`internalize_coding_bot` instead of calling `SelfCodingManager` directly so
that registration and ROI/error thresholds are handled automatically.
The pre-commit hook `self-coding-registration` (backed by
`tools/check_self_coding_registration.py`) scans all Python sources for classes
whose names end with `Bot` and verifies they are decorated with
`@self_coding_managed`. Run it before committing:

```bash
pre-commit run self-coding-registration --all-files
```

The hook exits with a non-zero status if any unmanaged bot class is found.

Additionally, `tools/find_unmanaged_bots.py` scans the repository for bot modules that lack the `@self_coding_managed` decorator or proper registration. Run it before pushing changes:

```bash
pre-commit run find-unmanaged-bots --all-files
```

CI fails if unmanaged bots are detected.

The pipeline also runs `tools/check_self_coding_registration.py` and
`tools/find_unmanaged_bots.py` to ensure every `*Bot` class is either decorated
with `@self_coding_managed` or registered via `internalize_coding_bot`. Run the
audits locally before committing:

```bash
python tools/check_self_coding_registration.py
python tools/find_unmanaged_bots.py
python tools/check_self_coding_decorator.py
```

These scripts exit with a non-zero status when unmanaged bots are present,
causing CI to fail.

The `check-coding-bot-decorators` hook (`tools/check_coding_bot_decorators.py`)
scans every bot module and fails if a bot class lacks the
`@self_coding_managed` decorator or the module omits an
`internalize_coding_bot` call. Run it locally when adding new bots:

```bash
pre-commit run check-coding-bot-decorators --all-files
```

CI invokes the same script via `make self-coding-check` so the build fails when
an unmanaged bot is introduced.

The repository also ships `self_coding_audit.py`, which scans all Python files
for classes ending in `Bot` without `@self_coding_managed`. The audit runs as a
standalone pre-commit hook and CI job so missing decorators appear as their own
check. Run it locally with:

```bash
pre-commit run self-coding-audit --all-files
```

For additional assurance, the test suite includes
`tests/test_self_coding_compliance.py`, which executes the same scan during
`pytest`. The test fails if any `*_bot.py` module defines a bot class without
`@self_coding_managed`, ensuring new bots remain self-coding compliant.


## Helper generation wrappers

Calls to `SelfCodingEngine.generate_helper` must go through
`manager_generate_helper` so a `SelfCodingManager` context is established. A
pre-commit hook enforces this rule:

```bash
pre-commit run check-engine-generate-helper --all-files
```

Run the hook before committing to ensure no direct `engine.generate_helper`
calls slip into the codebase.

To run the same validation alongside the unmanaged bot scan, use the combined
Makefile target:

```bash
make self-coding-check
```

It executes `tools/check_self_coding_usage.py` and
`tools/find_unmanaged_bots.py`, failing if any unwrapped helper calls or
unmanaged bots are detected. CI runs this target automatically.

## Helper caller decoration

Any class that invokes `manager_generate_helper` or `generate_helper` must be
decorated with `@self_coding_managed`.  The `check-self-coding-decorator` hook
verifies this requirement:

```bash
pre-commit run check-self-coding-decorator --all-files
```

CI executes the same script, so missing decorators will cause the build to
fail.

## Patch provenance

Commits that modify self-coding infrastructure (`self_coding_manager.py`,
`self_coding_engine.py`, `quick_fix_engine.py`, or `coding_bot_interface.py`)
must include a `patch <id>` tag and provenance metadata. Install the
`commit-msg` hook to enforce this policy by creating a symlink from the
repository root:

```bash
ln -s ../../scripts/check_patch_provenance.py .git/hooks/commit-msg
```

The hook rejects commits touching these files when the tag is missing or when
the self-coding manager fails to provide provenance via the
`PATCH_PROVENANCE_FILE` environment variable.

In continuous integration environments where hooks may be skipped, run the
verification script:

```bash
python scripts/check_patch_provenance.py --ci
```

The script records `untracked_commit` events through `MutationLogger` so the
`EvolutionOrchestrator` can trigger rollback if manual commits slip through.

## Stripe integration

To centralize billing logic and API configuration, the `stripe` Python package
must only be imported by `stripe_billing_router.py`. Other modules should rely
on the router's helpers rather than interacting with Stripe directly. The
`check-stripe-imports` pre-commit hook enforces this restriction. The repository
also guards against accidental exposure of live credentials. The
`forbid-stripe-keys` hook scans all text files for strings resembling live
Stripe keys (for example, strings beginning with `sk` or `pk` that resemble live or test keys) or hard-coded Stripe API endpoints
such as `api.stripe[dot]com`.

Before submitting a pull request, run `pre-commit run --all-files` to execute
this and other checks locally.
