"""keep the error of a failed message

Revision ID: 0002
Revises: 0001
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("messages", sa.Column("error", sa.Text(), nullable=True))
    op.create_check_constraint(
        op.f("ck_messages_failed_has_error"),
        "messages",
        "(status = 'failed') = (error IS NOT NULL)",
    )


def downgrade() -> None:
    op.drop_constraint(op.f("ck_messages_failed_has_error"), "messages")
    op.drop_column("messages", "error")
