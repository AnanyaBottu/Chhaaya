"""keep an ASHA's answer out of search until a person approves it

Revision ID: 0008
Revises: 0007
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Answers stored before this revision were never reviewed, so start unapproved.
    op.add_column(
        "kb_chunks",
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("kb_chunks", "approved_at")
