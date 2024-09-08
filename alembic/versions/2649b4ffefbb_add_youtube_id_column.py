"""Add youtube_id column

Revision ID: 2649b4ffefbb
Revises: 0d91d8241c4a
Create Date: 2024-09-08 21:53:09.295037

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2649b4ffefbb'
down_revision: Union[str, None] = '0d91d8241c4a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.add_column('videos', sa.Column('youtube_id', sa.String(), nullable=True))


def downgrade():
    op.drop_column('videos', 'youtube_id')

