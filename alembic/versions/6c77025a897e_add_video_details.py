"""add video details

Revision ID: 6c77025a897e
Revises: 05be45ae8546
Create Date: 2024-09-06 01:01:14.321460

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6c77025a897e'
down_revision: Union[str, None] = '05be45ae8546'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade():
    op.add_column('videos', sa.Column('account_name', sa.String(), nullable=True))
    op.add_column('videos', sa.Column('game_mode', sa.String(), nullable=True))
    op.add_column('videos', sa.Column('weapon', sa.String(), nullable=True))
    op.add_column('videos', sa.Column('map_name', sa.String(), nullable=True))

def downgrade():
    op.drop_column('videos', 'map_name')
    op.drop_column('videos', 'weapon')
    op.drop_column('videos', 'game_mode')
    op.drop_column('videos', 'account_name')