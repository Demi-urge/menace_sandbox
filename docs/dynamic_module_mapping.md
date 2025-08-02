# Dynamic Module Mapping

`dynamic_module_mapper.py` builds a graph of interactions across the repository.
Every Python file is parsed so imports and function calls are discovered. When a
call uses a name imported from another module an edge is added between the
files, producing a `networkx` graph that models runtime dependencies. When
semantic matching is enabled the tool also compares docstrings and links modules
with similar descriptions.

Modules are clustered using HDBSCAN when available or community detection from
`networkx` otherwise. Pass `--algorithm hdbscan` to opt into HDBSCAN; the
package is optional so the tool falls back to community detection if it is not
installed. The resulting mapping is written to JSON with `build_module_map()`.

To generate a module map manually run the unified helper:

```bash
python scripts/generate_module_map.py
```

The old `scripts/build_module_map.py` helper has been removed after merging
its functionality here.

The helper accepts several flags:

- `--algorithm {greedy,label,hdbscan}` – choose the clustering algorithm.
- `--threshold <float>` – similarity threshold used by semantic mode.
- `--semantic` – enable docstring matching for additional edges.
- `--exclude` – glob patterns of directories to ignore when scanning.

Set `SANDBOX_AUTO_MAP=1` to have `sandbox_runner` build the map
automatically on startup. The file is written to
`SANDBOX_DATA_DIR/module_map.json` and loaded for ROI aggregation. Use
`SANDBOX_REFRESH_MODULE_MAP=1` to rebuild the map before sandbox cycles.
Enable docstring-based linking with `SANDBOX_SEMANTIC_MODULES=1`.
Provide comma-separated patterns in `SANDBOX_EXCLUDE_DIRS` to exclude
directories from the generated map.
Run `sandbox_runner` with `--refresh-module-map` to refresh the mapping at
startup and monitor patches for new modules automatically.

### Semantic clustering example

With `--semantic` enabled the mapper groups modules sharing similar docstrings
even without direct imports. Running

```bash
python scripts/generate_module_map.py --semantic --threshold 0.2
```

will cluster any modules describing *"ROI prediction helpers"* together, even
if they call each other dynamically rather than via normal imports.

### Dynamic workflows

When `sandbox_runner` starts with `--dynamic-workflows` it builds
temporary workflows from the module groups returned by the mapper. The
feature is triggered automatically when the workflow database is empty
and can be tuned via the same options used for generating the module
map:

- `--module-algorithm` – clustering algorithm for grouping modules.
- `--module-threshold` – semantic similarity threshold.
- `--module-semantic` – enable docstring similarity.

For example the following command creates semantic module groups and
executes them as workflows:

```bash
python run_autonomous.py --dynamic-workflows --module-semantic \
  --module-threshold 0.25
```

The temporary workflows are discarded once explicit definitions are
stored in `workflows.db`.

### Automatic orphan detection

`sandbox_runner` automatically discovers modules that are not referenced by any
tests. The self‑test service loads `sandbox_data/orphan_modules.json` or
generates it using `discover_recursive_orphans` when missing and follows orphan
dependencies by default. Modules that pass their tests are
merged into `module_map.json` so future runs treat them like regular members of
their assigned groups. When the integration succeeds simple one-step workflows
are created for the new modules so they can be benchmarked immediately.
If a discovered module belongs to an existing workflow, the sandbox will attempt to merge it into that sequence automatically.
Set `SELF_TEST_RECURSIVE_ORPHANS=0` or `SANDBOX_RECURSIVE_ORPHANS=0`, or pass `--no-recursive-orphans`, to disable dependency scanning when building the list.

Example workflow:

```bash
python run_autonomous.py --discover-orphans --include-orphans
```

The command scans for orphan modules and follows their dependencies by default
using `sandbox_runner.discover_recursive_orphans`, runs them with the standard
unit tests and updates the module map once they succeed.  Disable dependency
traversal with `--no-recursive-orphans` or by setting
`SANDBOX_RECURSIVE_ORPHANS=0`.

### Recursive isolated modules

Isolated modules returned by `discover_isolated_modules` are loaded when
`--auto-include-isolated` is supplied or `SANDBOX_AUTO_INCLUDE_ISOLATED=1` is
set and are processed recursively by default. These isolated modules are tested
along with any dependencies and merged into `module_map.json` once they pass so
later runs treat them as normal members of their assigned groups. Simple
workflows are generated for the new modules and existing flows are updated via
`try_integrate_into_workflows`. Dependency traversal is enabled by default
through `SELF_TEST_RECURSIVE_ISOLATED=1` and `SANDBOX_RECURSIVE_ISOLATED=1`; set
`SELF_TEST_RECURSIVE_ISOLATED=0`, `SANDBOX_RECURSIVE_ISOLATED=0` or use
`--no-recursive-isolated` when running the self tests manually to disable this
behaviour. Redundant modules flagged by `orphan_analyzer.analyze_redundancy`
are skipped during integration.

