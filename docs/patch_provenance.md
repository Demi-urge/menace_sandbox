# Patch Provenance

The patch provenance utilities expose information about patches stored in
`PatchHistoryDB`.

## CLI

Use `patch_provenance_cli.py` to inspect patch records:

```bash
python patch_provenance_cli.py list
python patch_provenance_cli.py show 1
python patch_provenance_cli.py search vec123
```

`search` will first look for matching vector IDs in provenance records and
fallback to keyword search over patch descriptions and filenames.

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
