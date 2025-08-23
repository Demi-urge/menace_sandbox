# Contributor Guidelines

## SQLite connections

Use `DBRouter` for all database access. Direct calls to `sqlite3.connect` are not permitted outside approved scaffolding scripts (`scripts/new_db.py`, `scripts/new_db_template.py`, `scripts/scaffold_db.py`, and `scripts/new_vector_module.py`). CI runs `scripts/check_sqlite_connections.py` via pre-commit to enforce this rule.
