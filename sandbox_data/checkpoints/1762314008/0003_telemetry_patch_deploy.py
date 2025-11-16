"""add patch and deploy ids to telemetry

Revision ID: 0003
Revises: 0002
Create Date: 2025-06-25 00:00:00.000000
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '0003'
down_revision: Union[str, Sequence[str], None] = '0002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('telemetry', sa.Column('patch_id', sa.Integer(), nullable=True))
    op.add_column('telemetry', sa.Column('deploy_id', sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column('telemetry', 'deploy_id')
    op.drop_column('telemetry', 'patch_id')

