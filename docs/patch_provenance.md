# Patch Provenance

The patch provenance utilities expose information about patches stored in
`PatchHistoryDB`.  They surface the vector ancestry recorded in the
`patch_ancestry` table so that developers can see which vectors influenced a
patch and their influence scores.

## CLI

Use `tools/patch_provenance_cli.py` to inspect patch records:

```bash
python tools/patch_provenance_cli.py list
python tools/patch_provenance_cli.py show 1
python tools/patch_provenance_cli.py chain 1
python tools/patch_provenance_cli.py search vec123
python tools/patch_provenance_cli.py search --hash <code_hash>
python tools/patch_provenance_cli.py search --license MIT
python tools/patch_provenance_cli.py search --semantic-alert malware
python tools/patch_provenance_cli.py search --license MIT --semantic-alert malware
```

`list` displays recent patches with their IDs.  `show` prints the ancestry
details for a patch including the vectors, their influence scores and any
semantic alerts while `chain` returns only the ancestry chain for a patch.
`search` performs a reverse lookup either by vector ID (default) or by code
snippet hash when ``--hash`` is supplied.  It can also filter patches by
detected license, semantic alert, or both using ``--license`` and
``--semantic-alert``. These options mirror the `license` and
`semantic_alert` parameters supported by the REST service.

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
- `GET /patches/<patch_id>` – show a patch, its provenance and ancestry chain
- `GET /vectors/<vector_id>` – find patches influenced by a vector ID
- `GET /search?q=<term>` – search patches by vector ID or keyword
- `GET /patches?license=<name>&semantic_alert=<alert>` – filter patches by
  detected license and/or semantic alert

`/vectors/<vector_id>` and `/search` support pagination via `limit` and
`offset` query parameters.  An optional `index_hint` parameter can force a
specific SQLite index such as `idx_ancestry_vector` or `idx_patch_filename`.
For example:

```bash
curl "http://localhost:5000/vectors/vec123?limit=50&offset=100&index_hint=idx_ancestry_vector"
```

The service uses the same `PATCH_HISTORY_DB_PATH` environment variable to
locate the database.

## Python helpers

Two helpers are available for programmatic access:

```python
from patch_provenance import (
    search_patches_by_vector,
    search_patches,
    get_patch_provenance,
)

search_patches_by_vector("vec123")  # -> list of patches ordered by influence
search_patches(license="MIT", semantic_alert="malware")  # -> filtered patches
get_patch_provenance(1)              # -> vectors for a patch
```

The underlying :class:`PatchHistoryDB` also exposes
``get_ancestry(patch_id)``, ``get_ancestry_chain(patch_id)``,
``find_patches_by_vector(vector_id)``, and ``find_patches_by_hash(code_hash)``
for direct database access.

These functions integrate with `lineage_tracker.py` and `mutation_lineage.py`
to provide vector ancestry alongside existing lineage data.
