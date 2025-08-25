"""make content_hash non-null"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from db_dedup import compute_content_hash

revision: str = "0012"
down_revision: Union[str, Sequence[str], None] = "0011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLES: dict[str, list[str]] = {
    "bots": [
        "name",
        "type",
        "tasks",
        "dependencies",
        "purpose",
        "tags",
        "toolchain",
        "version",
    ],
    "workflows": [
        "workflow",
        "action_chains",
        "argument_strings",
        "title",
        "description",
        "task_sequence",
    ],
    "enhancements": [
        "idea",
        "rationale",
        "summary",
        "before_code",
        "after_code",
        "description",
    ],
    "errors": [
        "type",
        "description",
        "resolution",
    ],
}


def _compute_hash(row: dict[str, str], fields: list[str]) -> str:
    payload = {f: row.get(f) for f in fields}
    return compute_content_hash(payload)


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = set(inspector.get_table_names())
    for table, fields in TABLES.items():
        if table not in existing:
            continue
        pk_cols = inspector.get_pk_constraint(table)["constrained_columns"]
        if not pk_cols:
            continue
        pk = pk_cols[0]
        rows = (
            bind.execute(
                sa.text(
                    "SELECT {pk}, {cols} FROM {table} WHERE content_hash IS NULL".format(
                        pk=pk,
                        cols=", ".join(fields),
                        table=table,
                    )
                )
            )
            .mappings()
            .all()
        )
        for row in rows:
            content_hash = _compute_hash(row, fields)
            bind.execute(
                sa.text(
                    f"UPDATE {table} SET content_hash=:h WHERE {pk}=:id"
                ),
                {"h": content_hash, "id": row[pk]},
            )
        with op.batch_alter_table(table, recreate="always") as batch:
            batch.alter_column("content_hash", existing_type=sa.Text(), nullable=False)


def downgrade() -> None:
    for table in TABLES:
        with op.batch_alter_table(table, recreate="always") as batch:
            batch.alter_column("content_hash", existing_type=sa.Text(), nullable=True)
