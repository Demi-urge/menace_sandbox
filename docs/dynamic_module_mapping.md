# Dynamic Module Mapping

`module_mapper.py` builds a graph of module interactions across the repository. Each Python file is parsed with `ast` so both imports and call expressions are discovered. When a call uses a name imported from another module an edge is added between the files. The resulting `networkx` graph represents import relationships as well as runtime dependencies.

`cluster_modules()` then groups nodes using HDBSCAN if installed or falls back to the greedy modularity algorithm from `networkx`. The mapping is written to JSON with `save_module_map()`.

To generate a module map manually run:

```bash
python scripts/generate_module_map.py --root . --output sandbox_data/module_map.json
```

Additional options allow you to choose the clustering algorithm, adjust the threshold and enable semantic docstring matching.

`SANDBOX_AUTODISCOVER_MODULES=1` tells `sandbox_runner` to invoke this process automatically when the sandbox starts. The map is stored in `SANDBOX_DATA_DIR/module_map.json` and used to attribute ROI metrics per module group.
`SANDBOX_REFRESH_MODULE_MAP=1` forces regeneration of the map even when the file already exists.

