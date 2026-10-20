"""Review ASHA answers before later patients can be shown them.

An ASHA's answer to a question case is saved unapproved, and search skips it
until a person approves it: nothing else checks that it holds no patient's
name or personal details. A rejected answer is deleted.

    uv run python -m chhaaya.review               # list answers awaiting review
    uv run python -m chhaaya.review approve <id>  # make the answer searchable
    uv run python -m chhaaya.review reject <id>   # delete it
"""

import sys

from sqlalchemy import Engine, create_engine, delete, func, select, update

from chhaaya.db import KbChunk, sqlalchemy_url
from chhaaya.settings import Settings

UNREVIEWED = (KbChunk.source_type == "health_worker_answer") & (
    KbChunk.approved_at.is_(None)
)


def pending(engine: Engine) -> list[tuple[int, str]]:
    """The id and text of every answer awaiting review, oldest first."""
    statement = select(KbChunk.id, KbChunk.text).where(UNREVIEWED).order_by(KbChunk.id)
    with engine.connect() as conn:
        return [(row.id, row.text) for row in conn.execute(statement)]


def approve(engine: Engine, chunk_id: int) -> None:
    decide(engine, update(KbChunk).values(approved_at=func.now()), chunk_id)


def reject(engine: Engine, chunk_id: int) -> None:
    decide(engine, delete(KbChunk), chunk_id)


def decide(engine: Engine, statement, chunk_id: int) -> None:
    with engine.begin() as conn:
        result = conn.execute(statement.where(KbChunk.id == chunk_id, UNREVIEWED))
        if result.rowcount != 1:
            raise LookupError(f"no ASHA answer {chunk_id} is awaiting review")


def main() -> None:
    engine = create_engine(sqlalchemy_url(Settings().database_url))
    match sys.argv[1:]:
        case []:
            for chunk_id, text in pending(engine):
                print(f"{chunk_id}: {text}")
        case ["approve", chunk_id]:
            approve(engine, int(chunk_id))
        case ["reject", chunk_id]:
            reject(engine, int(chunk_id))
        case _:
            sys.exit(__doc__)


if __name__ == "__main__":
    main()
