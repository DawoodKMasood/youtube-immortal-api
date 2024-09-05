"""create videos table

Revision ID: 05be45ae8546
Revises: 
Create Date: 2024-09-06 00:51:11.932778

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '05be45ae8546'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.create_table(
        'videos',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('filename', sa.String(), nullable=True),
        sa.Column('status', sa.Enum('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED', name='videostatus'), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_videos_filename'), 'videos', ['filename'], unique=False)
    op.create_index(op.f('ix_videos_id'), 'videos', ['id'], unique=False)

def downgrade():
    op.drop_index(op.f('ix_videos_id'), table_name='videos')
    op.drop_index(op.f('ix_videos_filename'), table_name='videos')
    op.drop_table('videos')
