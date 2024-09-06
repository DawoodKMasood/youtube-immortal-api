"""Add current_stage

Revision ID: c6f8366e162e
Revises: 3f2540e75383
Create Date: 2024-09-06 07:26:33.126847

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c6f8366e162e'
down_revision: Union[str, None] = '3f2540e75383'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.add_column('videos', sa.Column('current_stage', sa.String(), nullable=True))


def downgrade():
    op.drop_column('videos', 'current_stage')