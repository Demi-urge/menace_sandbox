# Module Synergy Grapher

`module_synergy_grapher` builds a weighted graph that links modules based on direct
imports and shared dependencies, structural similarity and co-occurrence data from
workflows and historical synergy records.  The resulting graph is saved to
`sandbox_data/module_synergy_graph.json` for later queries.

## Building

```bash
python module_synergy_grapher.py --build [--auto-tune] [--config config.toml]
```

The graph is persisted to `sandbox_data/module_synergy_graph.json` and can be
rebuilt via `make synergy-graph` in automation contexts.

## Caching

To speed up subsequent runs, AST-derived identifiers and docstring embeddings
for each module are cached in `sandbox_data/synergy_cache.json`.  Each cache
entry stores the source file's modification time and a SHA256 hash.  During
`build_graph` these values are compared against the current file; modules with
unchanged metadata reuse cached details while others are recomputed.  Deleting
or touching the source file invalidates its cache entry, and passing
`--no-cache` rebuilds all entries from scratch.

## Querying

```bash
python module_synergy_grapher.py --cluster <module> --threshold 0.8
```

The `--threshold` flag filters edges by weight when expanding the cluster.

## Example `get_synergy_cluster` Output

After building the graph a module's neighbourhood can be inspected from the CLI:

```bash
$ python module_synergy_grapher.py --cluster a --threshold 0.5
a
b
```

The function form mirrors the CLI:

```python
from module_synergy_grapher import get_synergy_cluster
get_synergy_cluster("a", threshold=0.5)
# {'a', 'b'}
```

## CLI Options

`module_synergy_grapher.py` exposes the following commands:

* `--build` – regenerate the synergy graph for the current repository.
* `--cluster MODULE` – print synergistic neighbours for `MODULE`.
* `--threshold FLOAT` – minimum cumulative synergy required for inclusion (default `0.7`).
* `--config PATH` – JSON/TOML file providing coefficient overrides.
* `--no-cache` – recompute AST info and embeddings ignoring cached results.
* `--embed-workers N` – number of threads used when fetching embeddings.
* `--auto-tune` – learn coefficient weights from `synergy_history.db` before rebuilding.

## Configuration

The `ModuleSynergyGrapher` constructor accepts either a `coefficients` mapping
or a JSON/TOML `config` file to adjust how each signal contributes to an edge:

```python
from module_synergy_grapher import ModuleSynergyGrapher

grapher = ModuleSynergyGrapher(config="weights.json")
```

The CLI also accepts `--config` to load the same style of configuration file
when building the graph.

The optional `embedding_threshold` parameter controls the minimum cosine
similarity between module docstrings required before an embedding edge is
added to the graph.

Use the :meth:`save` method to serialise the graph in JSON or pickle format:

```python
graph = grapher.build_graph(".")
grapher.save(graph, format="pickle")
```

## Data Sources

Edges combine four heuristics:

* **Imports and shared dependencies** – direct imports and overlap in imported
  modules derived from the static import graph.
* **Shared identifiers** – Jaccard similarity of variable, function and class
  names extracted from each module's AST.
* **Workflow co-occurrence** – pairs of modules that appear together in
  `workflows.db` or in historical synergy records.
* **Docstring embeddings** – cosine similarity of module docstring embeddings
  stored in `sandbox_data/module_doc_embeddings.*`.

## Graph Metrics

The resulting graph is weighted and directed. Each edge's `weight` captures the
combined import, structural, co-occurrence and embedding signals. Standard
NetworkX routines can then surface relationships:

* **Weighted degree** – `graph.degree(node, weight="weight")` highlights how
  strongly a module is connected overall.
* **In/out degree** – `graph.in_degree(node, weight="weight")` /
  `graph.out_degree(node, weight="weight")` distinguish inbound and outbound
  synergy.
* **Clustering coefficient** – `nx.clustering(graph.to_undirected(), node, weight="weight")`
  measures how tightly neighbours interact.
* **PageRank** – `nx.pagerank(graph, weight="weight")` surfaces globally
  influential modules.

## Limitations

* Only static imports are analysed; dynamic or runtime imports are ignored.
* Embedding generation relies on optional `sentence_transformers` models and may
  be slow or unavailable in minimal environments.
* Co-occurrence signals require populated `workflows.db` or
  `synergy_history.db`; if these databases are absent the corresponding weights
  are zero.


