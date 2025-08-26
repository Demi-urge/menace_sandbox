# Contributor Guidelines

## SQLite connections

Use `DBRouter` for all database access. Direct calls to `sqlite3.connect` are not
permitted outside approved scaffolding scripts
(`scripts/new_db.py`, `scripts/new_db_template.py`, `scripts/scaffold_db.py`, and
`scripts/new_vector_module.py`).  In the rare case a raw connection is required,
include explicit audit logging and add the module to
`tests/approved_sqlite3_usage.txt` with a justification.

To avoid accidental direct connections, CI runs
`tests/test_db_router_enforcement.py` and a pre-commit hook which executes
`scripts/check_sqlite_connections.py` against all Python sources. These checks
fail the build if an unapproved `sqlite3.connect` call is detected. Legitimate
new uses must be added to `tests/approved_sqlite3_usage.txt` together with an
explanation for the exemption.
