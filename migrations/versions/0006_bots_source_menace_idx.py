"""add source_menace_id column and index for bots"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0006"
down_revision: Union[str, Sequence[str], None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "bots" not in inspector.get_table_names():
        return
    cols = {c["name"] for c in inspector.get_columns("bots")}
    if "source_menace_id" not in cols:
        op.add_column(
            "bots",
            sa.Column("source_menace_id", sa.Text(), nullable=False, server_default=""),
        )
        op.alter_column("bots", "source_menace_id", server_default=None)
    idxs = {i["name"] for i in inspector.get_indexes("bots")}
    if "ix_bots_source_menace_id" in idxs:
        op.drop_index("ix_bots_source_menace_id", table_name="bots")
    if "idx_bots_source_menace_id" not in idxs:
        op.create_index(
            "idx_bots_source_menace_id",
            "bots",
            ["source_menace_id"],
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "bots" not in inspector.get_table_names():
        return
    idxs = {i["name"] for i in inspector.get_indexes("bots")}
    if "idx_bots_source_menace_id" in idxs:
        op.drop_index("idx_bots_source_menace_id", table_name="bots")
    cols = {c["name"] for c in inspector.get_columns("bots")}
    if "source_menace_id" in cols:
        op.drop_column("bots", "source_menace_id")
