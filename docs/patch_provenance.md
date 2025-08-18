# Patch Provenance

The patch provenance utilities expose information about patches stored in
`PatchHistoryDB`.  They surface the vector ancestry recorded in the
`patch_ancestry` table so that developers can see which vectors influenced a
patch and their influence scores.

## CLI

Use `patch_provenance_cli.py` to inspect patch records:

```bash
python patch_provenance_cli.py list
python patch_provenance_cli.py show 1
python patch_provenance_cli.py search vec123
```

`search` looks for matching vector IDs in the ancestry records and orders the
results by influence score.  If no vectors match the term it falls back to a
keyword search over patch descriptions and filenames.  `show` displays the
vectors influencing a patch ordered by their influence.

Set the `PATCH_HISTORY_DB_PATH` environment variable to point to a specific
SQLite database.

## REST Service

`patch_provenance_service.py` provides the same queries over HTTP using
Flask.  Start the service with:

```bash
python patch_provenance_service.py
```

Endpoints:

- `GET /patches` – list patches
- `GET /patches/<patch_id>` – show a patch and its provenance
- `GET /search?q=<term>` – search by vector ID or keyword

The service uses the same `PATCH_HISTORY_DB_PATH` environment variable to
locate the database.

## Python helpers

Two helpers are available for programmatic access:

```python
from patch_provenance import search_patches_by_vector, get_patch_provenance

search_patches_by_vector("vec123")  # -> list of patches ordered by influence
get_patch_provenance(1)              # -> vectors for a patch
```

These functions integrate with `lineage_tracker.py` and `mutation_lineage.py`
to provide vector ancestry alongside existing lineage data.
