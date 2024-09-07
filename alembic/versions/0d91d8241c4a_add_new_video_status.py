"""Add new video status

Revision ID: 0d91d8241c4a
Revises: 0dac890b0969
Create Date: 2024-09-08 03:08:40.124730

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0d91d8241c4a'
down_revision: Union[str, None] = '0dac890b0969'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # Use PostgreSQL-specific commands to alter the enum type
    op.execute("ALTER TYPE videostatus ADD VALUE IF NOT EXISTS 'UPLOADED'")

def downgrade():
    # Note: PostgreSQL doesn't support removing enum values
    # We'll leave this empty as we can't easily revert this change
    pass
