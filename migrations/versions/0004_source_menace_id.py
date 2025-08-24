"""add source_menace_id to shared tables"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '0004'
down_revision: Union[str, Sequence[str], None] = '0003'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

tables = [
    "bots",
    "code",
    "enhancements",
    "errors",
    "discrepancies",
    "workflow_summaries",
]


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = set(inspector.get_table_names())
    for table in tables:
        if table not in existing:
            continue
        op.add_column(
            table,
            sa.Column(
                "source_menace_id", sa.Text(), nullable=False, server_default=""
            ),
        )
        op.create_index(
            f"ix_{table}_source_menace_id", table, ["source_menace_id"]
        )
        op.alter_column(table, "source_menace_id", server_default=None)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = set(inspector.get_table_names())
    for table in tables:
        if table not in existing:
            continue
        op.drop_index(f"ix_{table}_source_menace_id", table_name=table)
        op.drop_column(table, "source_menace_id")
