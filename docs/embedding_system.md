# Embedding System

The `EmbeddableDBMixin` equips SQLite-backed databases with persistent vector
embeddings and similarity search. It maintains an Annoy or FAISS index on disk
alongside JSON metadata and lazily loads a `SentenceTransformer` model to
encode text. Subclasses provide a database connection and implement
`vector(record)` plus `iter_records()` to describe how records are embedded and
enumerated for backfilling.

## Configuration

`EmbeddableDBMixin` accepts several options when initialised:

- **backend** – either `"annoy"` (default) or `"faiss"` to choose the vector
  index implementation.
- **index_path** – file path where the index is stored.
- **metadata_path** – companion JSON storing id mappings and metadata.
- **model_name** – sentence-transformer used for encoding.
- **embedding_version** – version tag for stored vectors.

Embeddings are added via `add_embedding()` and queried with
`search_by_vector()`. `backfill_embeddings()` iterates over records lacking
vectors and persists new embeddings using the configured backend and paths.

## Databases Using the Mixin

Several databases leverage the mixin, each supplying its own index location and
record-to-vector logic:

| Database | Default index | Description |
| --- | --- | --- |
| `BotDB` | `bot_embeddings.index` | Embeds bot metadata for semantic lookup. |
| `WorkflowDB` | `workflow_embeddings.index` | Stores workflow representations for search. |
| `EnhancementDB` | `enhancement_embeddings.ann` | Indexes enhancement logs. |
| `ErrorDB` | `error_embeddings.index` | Tracks known errors and discrepancies. |
| `InfoDB` | `information_embeddings.index` | Embeds research items. |
| `InformationDB` | `information_embeddings.index` | Stores static information records. |

## Backfilling Embeddings

Use `scripts/backfill_embeddings.py` to generate embeddings for existing
records:

```bash
python scripts/backfill_embeddings.py --db bot workflow --backend annoy --batch-size 200
```

Omit `--db` to process all supported databases. `--backend` selects the vector
backend and `--batch-size` controls how many records are processed at once.

## Universal Retriever

`UniversalRetriever` performs cross-database similarity search by orchestrating
multiple mixin-backed stores. It collects the top candidates from each database
and assigns a confidence score to every match.

See [universal_retriever.md](universal_retriever.md) for the full API, scoring
formula and integration guide.

`DatabaseRouter.semantic_search` now delegates to the retriever so callers can
perform a single query across bots, workflows, errors, enhancements and
research items without dealing with individual databases. Each result includes
the originating database label, combined score and a short reason describing the
highest contributing metric.

## Service Layer

`vector_service` provides lightweight facades—`Retriever`, `ContextBuilder`, `PatchLogger` and `EmbeddingBackfill`—that wrap the embedding components with structured logging and Prometheus metrics. Other modules should depend on this layer instead of calling databases or retrievers directly. The service layer keeps instrumentation consistent and makes unit testing simple by allowing dependencies to be swapped or mocked.
