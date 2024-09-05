"""Add new VideoStatus values

Revision ID: 3f2540e75383
Revises: 671c52ceb157
Create Date: 2024-09-06 01:48:31.925711

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3f2540e75383'
down_revision: Union[str, None] = '671c52ceb157'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # Use PostgreSQL-specific commands to alter the enum type
    op.execute("ALTER TYPE videostatus ADD VALUE IF NOT EXISTS 'APPROVED'")
    op.execute("ALTER TYPE videostatus ADD VALUE IF NOT EXISTS 'REJECTED'")

def downgrade():
    # Note: PostgreSQL doesn't support removing enum values
    # We'll leave this empty as we can't easily revert this change
    pass