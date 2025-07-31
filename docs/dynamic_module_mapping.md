# Dynamic Module Mapping

`dynamic_module_mapper.py` builds a graph of interactions across the repository.
Every Python file is parsed so imports and function calls are discovered. When a
call uses a name imported from another module an edge is added between the
files, producing a `networkx` graph that models runtime dependencies. When
semantic matching is enabled the tool also compares docstrings and links modules
with similar descriptions.

Modules are clustered using HDBSCAN when available or community detection from
`networkx` otherwise. The resulting mapping is written to JSON with
`build_module_map()`.

To generate a module map manually run the unified helper:

```bash
python scripts/build_module_map.py
```

The legacy `scripts/generate_module_map.py` script remains as a thin alias.

The helper accepts several flags:

- `--algorithm {greedy,label}` – choose the community detection algorithm.
- `--threshold <float>` – similarity threshold used by semantic mode.
- `--semantic` – enable docstring matching for additional edges.

Set `SANDBOX_AUTODISCOVER_MODULES=1` to have `sandbox_runner` build the map
automatically on startup. The file is written to
`SANDBOX_DATA_DIR/module_map.json` and loaded for ROI aggregation. Use
`SANDBOX_REFRESH_MODULE_MAP=1` to force regeneration when the map already exists.

### Semantic clustering example

With `--semantic` enabled the mapper groups modules sharing similar docstrings
even without direct imports. Running

```bash
python scripts/build_module_map.py --semantic --threshold 0.2
```

will cluster any modules describing *"ROI prediction helpers"* together, even
if they call each other dynamically rather than via normal imports.

