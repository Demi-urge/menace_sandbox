"""add content_hash columns"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import hashlib
import json

revision: str = "0010"
down_revision: Union[str, Sequence[str], None] = "0009"
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
        "message",
        "type",
        "description",
        "resolution",
    ],
}


def _compute_hash(row: dict[str, str], fields: list[str]) -> str:
    payload = {f: row.get(f) for f in fields}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = set(inspector.get_table_names())
    for table, fields in TABLES.items():
        if table not in existing:
            continue
        cols = {c["name"] for c in inspector.get_columns(table)}
        if "content_hash" not in cols:
            op.add_column(table, sa.Column("content_hash", sa.Text(), nullable=True))
        idxs = {i["name"] for i in inspector.get_indexes(table)}
        idx_name = f"idx_{table}_content_hash"
        if idx_name not in idxs:
            op.create_index(idx_name, table, ["content_hash"], unique=True)
        pk_cols = inspector.get_pk_constraint(table)["constrained_columns"]
        if not pk_cols:
            continue
        pk = pk_cols[0]
        rows = (
            bind.execute(
                sa.text(
                    "SELECT {} , {} FROM {}".format(
                        pk, ", ".join(fields), table
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


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = set(inspector.get_table_names())
    for table in TABLES:
        if table not in existing:
            continue
        idx_name = f"idx_{table}_content_hash"
        idxs = {i["name"] for i in inspector.get_indexes(table)}
        if idx_name in idxs:
            op.drop_index(idx_name, table_name=table)
        cols = {c["name"] for c in inspector.get_columns(table)}
        if "content_hash" in cols:
            op.drop_column(table, "content_hash")

