"""Add thumbnail column to videos table

Revision ID: 671c52ceb157
Revises: 6c77025a897e
Create Date: 2024-09-06 01:39:02.217605

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '671c52ceb157'
down_revision: Union[str, None] = '6c77025a897e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.add_column('videos', sa.Column('thumbnail', sa.String(), nullable=True))


def downgrade():
    op.drop_column('videos', 'thumbnail')
