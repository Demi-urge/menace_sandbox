# Dynamic Path Router

The dynamic path router centralises runtime file lookups within the repository.
It exposes helpers for resolving files relative to the project root so scripts
and documentation remain portable across forks or nested clones.

## Runtime resolution

`dynamic_path_router.resolve_path` returns an absolute `pathlib.Path` for the
requested file.  The project root is determined at runtime using the following
order:

1. The `MENACE_ROOTS`/`SANDBOX_REPO_PATHS` environment variables (colon
   separated list of roots).
2. The `MENACE_ROOT` or `SANDBOX_REPO_PATH` environment variable.
3. `git rev-parse --show-toplevel`.
4. Searching parent directories for a `.git` folder.
5. Falling back to the directory containing `dynamic_path_router.py`.

If the direct lookup fails the router performs a recursive walk of the
repository to locate the file.

When the multi-root variables are present they take precedence over the single
root overrides.  Each entry is searched in order and the first matching root is
stored for reuse so subsequent lookups avoid repeated filesystem walks.

## Examples

Always call `resolve_path` instead of embedding string literals. The resolver
honours environment overrides and multi-root configurations.

### Override repository root

```bash
MENACE_ROOT=/alt/clone python - <<'PY'
from dynamic_path_router import resolve_path
print(resolve_path('sandbox_runner.py'))
PY
```

### Multi-root lookup

```bash
MENACE_ROOTS="/repo/main:/repo/fork" python - <<'PY'
from dynamic_path_router import resolve_path
print(resolve_path('sandbox_runner.py', repo_hint='/repo/fork'))
PY
```

## Additional helpers

- `get_project_root` and `get_project_roots` expose the discovered repository
  roots, optionally selecting one via the `repo_hint` parameter.
- `resolve_module_path` resolves dotted module names to their source file or
  package `__init__.py` across all registered roots.
- `resolve_dir` resolves and validates directory locations.

All resolutions and discovered roots are cached behind a thread-safe lock to
keep repeated lookups inexpensive.

## Caching

Successful resolutions are cached in-memory to avoid repeated scans.  Call
`dynamic_path_router.clear_cache()` when tests or tools create or remove files
and need fresh lookups.  `dynamic_path_router.list_files()` exposes the internal
cache mapping for debugging purposes.

When multiple roots are configured, pass a repository name or path via the
`repo_hint` argument to `get_project_root`/`resolve_path` to target a specific
repository.

## Migration tips

- Prefer `resolve_path` for all repository file references rather than
  hard-coded relative paths.
- In shell snippets, resolve script locations dynamically:
  ```bash
  python "$(python - <<'PY'
  from dynamic_path_router import resolve_path
  print(resolve_path('sandbox_runner.py'))
  PY
  )" --help
  ```
- When adding new files, update documentation and scripts to use
  `resolve_path` and clear the cache in tests if paths change.
- External tools can override the repository root by setting `MENACE_ROOT` or
  `SANDBOX_REPO_PATH`.

## Static path enforcement

A `pre-commit` hook named `check-static-paths` scans modified Python files for
string literals that end with `.py`. These paths **must** be wrapped with
`resolve_path` or `path_for_prompt` to remain portable across forks and
clones. The hook fails if an unwrapped `.py` path is found.

Run `pre-commit run check-static-paths --files <changed files>` locally before
committing to catch violations early. The checker can also be invoked directly
for ad-hoc scans:

```bash
python tools/check_static_paths.py $(git ls-files 'self_*' 'sandbox_runner/*.py')
```

The command exits with a non-zero status and lists offending literals when
violations are detected.
