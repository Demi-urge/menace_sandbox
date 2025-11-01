# Modality Registration

This guide describes how to add a new record modality to the vector service.

## Implement an Embeddable Database

Create a database class that mixes your existing SQLite logic with `EmbeddableDBMixin`.
The mixin handles storage for embedding vectors and provides similarity search helpers.
Override `vector` to convert records into embeddings and `iter_records` to stream
existing rows for backfilling.

```python
from menace_sandbox.embeddable_db_mixin import EmbeddableDBMixin
from experiment_history_db import ExperimentHistoryDB

class ExperimentDB(ExperimentHistoryDB, EmbeddableDBMixin):
    def __init__(self, path: str = "experiment_history.db"):
        ExperimentHistoryDB.__init__(self, path)
        EmbeddableDBMixin.__init__(
            self,
            index_path="experiment.ann",
            metadata_path="experiment.json",
            embedding_version=1,
        )

    def iter_records(self):
        cur = self.conn.execute(
            "SELECT rowid, variant || ' ' || roi FROM experiment_history"
        )
        for rid, text in cur:
            yield rid, text, "experiment"

    def vector(self, record: str) -> list[float]:
        return self.encode_text(record)
```

`vector` uses `encode_text` to produce a list of floats for each record. `iter_records`
yields `(record_id, record, kind)` tuples so `EmbeddingBackfill` can generate embeddings
for existing rows. Override `license_text` when non‑string records need license scanning.

## Register the Vectorizer

Implement a matching vectorizer and register both pieces with
`vector_service.registry.register_vectorizer` so the service can discover them.

```python
from vector_service.registry import register_vectorizer

register_vectorizer(
    "experiment",
    "experiment_vectorizer",
    "ExperimentVectorizer",
    db_module="experiment_history_db",
    db_class="ExperimentDB",
)
```

## Run the Backfill

Once registered, the standard backfill process handles the new modality:

```python
from vector_service import EmbeddingBackfill

EmbeddingBackfill().run(dbs=["experiment"])
```

`EmbeddingBackfill` initialises `ExperimentDB`, generates missing embeddings, and
persists them to an Annoy/FAISS index with JSON metadata.
