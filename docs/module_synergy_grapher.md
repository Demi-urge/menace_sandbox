# Module Synergy Grapher

`module_synergy_grapher` builds a weighted graph that links modules based on direct
imports and shared dependencies, structural similarity and co-occurrence data from
workflows and historical synergy records.  The resulting graph is saved to
`sandbox_data/module_synergy_graph.json` for later queries.

## Building

```bash
python module_synergy_grapher.py --build [--config config.toml]
```

The graph is persisted to `sandbox_data/module_synergy_graph.json` and can be
rebuilt via `make synergy-graph` in automation contexts.

## Querying

```bash
python module_synergy_grapher.py --cluster <module> --threshold 0.8
```

The `--threshold` flag filters edges by weight when expanding the cluster.

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

## Limitations

* Only static imports are analysed; dynamic or runtime imports are ignored.
* Embedding generation relies on optional `sentence_transformers` models and may
  be slow or unavailable in minimal environments.
* Co-occurrence signals require populated `workflows.db` or
  `synergy_history.db`; if these databases are absent the corresponding weights
  are zero.


