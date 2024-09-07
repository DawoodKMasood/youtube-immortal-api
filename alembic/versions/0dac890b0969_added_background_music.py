"""Added background_music

Revision ID: 0dac890b0969
Revises: c6f8366e162e
Create Date: 2024-09-07 06:41:33.786936

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0dac890b0969'
down_revision: Union[str, None] = 'c6f8366e162e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.add_column('videos', sa.Column('background_music', sa.String(), nullable=True))


def downgrade():
    op.drop_column('videos', 'background_music')