# Contributor Guidelines

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

## Stripe integration

To centralize billing logic and API configuration, the `stripe` Python package
must only be imported by `stripe_billing_router.py`. Other modules should rely
on the router's helpers rather than interacting with Stripe directly. The
`check-stripe-imports` pre-commit hook enforces this restriction. The repository
also guards against accidental exposure of live credentials. The
`forbid-stripe-keys` hook scans all text files for strings resembling live
Stripe keys (e.g., `sk_live_` or `pk_live_`) or hard-coded Stripe API endpoints
such as `api.stripe.com`.

Before submitting a pull request, run `pre-commit run --all-files` to execute
this and other checks locally.
