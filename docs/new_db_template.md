# New Database Template

The CLI provides a `new-db` command for quickly creating a new SQLite-backed
module. The generated module mixes in
[`EmbeddableDBMixin`](../embeddable_db_mixin.py) and sets up a basic
full-text-search schema derived from `sql_templates/create_fts.sql`.

For buffered writes to shared tables and background queue processing, see
[write_buffer.md](write_buffer.md).

```bash
python menace_cli.py new-db sessions
```

Running the above command creates `sessions_db.py` in the project root and
updates `__init__.py` to import the new database class. The scaffolded module
includes placeholder hooks for license detection and unsafe code pattern checks
so additional safety logic can be implemented later.
