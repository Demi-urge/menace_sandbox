# Database Migrations

Menace uses Alembic to manage schema changes. Run migrations with:

```bash
alembic upgrade head
```

The `DATABASE_URL` environment variable controls which database is upgraded.

When updating an existing installation, run the migrations after pulling the
latest code:

```bash
alembic upgrade head
```

This will create any missing tables such as those defined in
`menace.databases.MenaceDB`. As of version 0.2 a `memory_embeddings` table is
created to store vector representations of memory entries for semantic search.
Databases that inherit from `EmbeddableDBMixin` (bots, workflows, errors,
enhancements and research info) now also maintain a shared `embeddings` table
for vector search.  After migrating, run `backfill_embeddings()` on each
database to populate vectors.

### Content hash column

Version 0.3 adds a `content_hash` column to the `bots`, `workflows`,
`enhancements`, and `errors` tables for deduplication. New installations create
the column automatically. For existing databases run the migration or add the
column manually:

```sql
ALTER TABLE bots ADD COLUMN content_hash TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS idx_bots_content_hash ON bots(content_hash);
ALTER TABLE workflows ADD COLUMN content_hash TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS idx_workflows_content_hash ON workflows(content_hash);
ALTER TABLE enhancements ADD COLUMN content_hash TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS idx_enhancements_content_hash ON enhancements(content_hash);
ALTER TABLE errors ADD COLUMN content_hash TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS idx_errors_content_hash ON errors(content_hash);
```

Re-run `alembic upgrade head` after adding the columns so Alembic can create any
missing indices.
