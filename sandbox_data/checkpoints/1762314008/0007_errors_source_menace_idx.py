"""add source_menace_id column and index for errors"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0007"
down_revision: Union[str, Sequence[str], None] = "0006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "errors" not in inspector.get_table_names():
        return
    cols = {c["name"] for c in inspector.get_columns("errors")}
    if "source_menace_id" not in cols:
        op.add_column(
            "errors",
            sa.Column("source_menace_id", sa.Text(), nullable=False, server_default=""),
        )
        op.alter_column("errors", "source_menace_id", server_default=None)
    idxs = {i["name"] for i in inspector.get_indexes("errors")}
    if "ix_errors_source_menace_id" in idxs:
        op.drop_index("ix_errors_source_menace_id", table_name="errors")
    if "idx_errors_source_menace_id" not in idxs:
        op.create_index("idx_errors_source_menace_id", "errors", ["source_menace_id"])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "errors" not in inspector.get_table_names():
        return
    idxs = {i["name"] for i in inspector.get_indexes("errors")}
    if "idx_errors_source_menace_id" in idxs:
        op.drop_index("idx_errors_source_menace_id", table_name="errors")
    cols = {c["name"] for c in inspector.get_columns("errors")}
    if "source_menace_id" in cols:
        op.drop_column("errors", "source_menace_id")
