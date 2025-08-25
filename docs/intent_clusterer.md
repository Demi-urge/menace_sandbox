# Intent Clusterer

`IntentClusterer` builds a lightweight semantic index of Python modules by
collecting docstrings, comments and symbol names.  Embeddings are stored via a
`UniversalRetriever`-compatible backend so other components can discover modules
related to a natural language goal.  Modules are assigned to intent clusters
using a soft‑clustering strategy so they may belong to multiple clusters when
similarity to several centroids exceeds a configurable threshold.  The
clusterer underpins several parts of the sandbox self‑improvement loop,
including the `self_improvement_engine` and `workflow_evolution_bot`.  The class
can be imported either directly or via the package root:

```python
from menace import IntentClusterer
```

## Indexing modules

```python
from pathlib import Path
from intent_clusterer import IntentClusterer
from universal_retriever import UniversalRetriever

retriever = UniversalRetriever(...)
clusterer = IntentClusterer(retriever)
module_paths = [
    Path("self_improvement_engine.py"),
    Path("bot_creation_bot.py"),
]
clusterer.index_modules(module_paths)
```

## Indexing a repository

To embed an entire repository and persist the vectors, call
``index_repository`` with the repository root.  Synergy clusters derived from
``sandbox_data/module_map.json`` (or ``ModuleSynergyGrapher`` when available)
are embedded automatically and can be queried alongside individual modules.

```python
from pathlib import Path
from menace import IntentClusterer

clusterer = IntentClusterer()
clusterer.index_repository(Path("/path/to/repo"))
```

## Retrieving intent results

Retrieve modules or clusters that relate to a textual prompt:

```python
mods = clusterer.find_modules_related_to("update configuration", top_k=5)
clusters = clusterer.find_clusters_related_to("update configuration", top_k=5)
for match in mods:
    print("module", match["path"], match["score"])
for match in clusters:
    print("cluster", match["path"], match.get("members"), match["score"])
```

Both functions return dictionaries describing the best matches.  For more
detailed similarity information including related cluster identifiers use
`query`:

```python
matches = clusterer.query("update configuration")
for m in matches:
    print(m.path, m.cluster_ids, m.similarity)
```

`query` yields `IntentMatch` objects with module paths, similarity scores and
any related cluster identifiers.
