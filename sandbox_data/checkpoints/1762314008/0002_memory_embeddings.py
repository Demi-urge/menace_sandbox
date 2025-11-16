"""add table for memory entry embeddings"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '0002'
down_revision: Union[str, Sequence[str], None] = '0001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'memory_embeddings',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('key', sa.String(), nullable=True),
        sa.Column('data', sa.Text(), nullable=True),
        sa.Column('version', sa.Integer(), nullable=True),
        sa.Column('tags', sa.Text(), nullable=True),
        sa.Column('ts', sa.String(), nullable=True),
        sa.Column('embedding', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('memory_embeddings')

