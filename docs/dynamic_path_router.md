# Dynamic Path Router

The dynamic path router centralises runtime file lookups within the repository.
It exposes helpers for resolving files relative to the project root so scripts
and documentation remain portable across forks or nested clones.

## Runtime resolution

`dynamic_path_router.resolve_path` returns an absolute `pathlib.Path` for the
requested file.  The project root is determined at runtime using the following
order:

1. The `MENACE_ROOT` or `SANDBOX_REPO_PATH` environment variable.
2. `git rev-parse --show-toplevel`.
3. Searching parent directories for a `.git` folder.
4. Falling back to the directory containing `dynamic_path_router.py`.

If the direct lookup fails the router performs a recursive walk of the
repository to locate the file.

## Caching

Successful resolutions are cached in-memory to avoid repeated scans.  Call
`dynamic_path_router.clear_cache()` when tests or tools create or remove files
and need fresh lookups.  `dynamic_path_router.list_files()` exposes the internal
cache mapping for debugging purposes.

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
