# Module Synergy Grapher

`module_synergy_grapher` builds a weighted graph that links modules based on direct
imports and shared dependencies, structural similarity and co-occurrence data from
workflows and historical synergy records.  The resulting graph is saved to
`sandbox_data/module_synergy_graph.json` for later queries.

## Building

```bash
python module_synergy_grapher.py build <root>
```

Use `--out` to choose a custom output path.

## Querying

```bash
python module_synergy_grapher.py cluster <module> --threshold 0.8 --path sandbox_data/module_synergy_graph.json
```

The `--threshold` flag filters edges by weight when expanding the cluster, and
`--path` points to an alternate graph file.

## Configuration

The `ModuleSynergyGrapher` constructor accepts a `coefficients` mapping to adjust
how each signal contributes to an edge:

```python
from module_synergy_grapher import ModuleSynergyGrapher

grapher = ModuleSynergyGrapher(
    coefficients={"import": 1.0, "structure": 0.5, "cooccurrence": 1.0}
)
```

Use the :meth:`save` method to serialise the graph in JSON or pickle format:

```python
graph = grapher.build_graph(".")
grapher.save(graph, format="pickle")
```


