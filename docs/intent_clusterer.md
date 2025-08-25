# Intent Clusterer

The `IntentClusterer` indexes Python modules by capturing docstrings,
comments and symbol names, storing embeddings via a retriever backend.
It can then answer semantic queries and surface related modules or
clusters.

## Indexing a repository

```python
from intent_clusterer import IntentClusterer
from universal_retriever import UniversalRetriever

retriever = UniversalRetriever(...)
clusterer = IntentClusterer(retriever)
clusterer.index_repository("/path/to/repo")
```

## Querying intents

```python
matches = clusterer.query("update configuration", top_k=5)
for match in matches:
    print(match.path, match.cluster_ids, match.similarity)
```

`query` returns `IntentMatch` objects with module paths, similarity
scores and any related cluster identifiers.
