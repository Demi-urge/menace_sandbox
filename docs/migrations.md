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
