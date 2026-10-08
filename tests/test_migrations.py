from uuid import uuid4

import pytest
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from conftest import migrate
from sqlalchemy import Engine, inspect, text
from sqlalchemy.exc import IntegrityError

from chhaaya.db import Base


def add_patient(engine: Engine) -> int:
    with engine.begin() as conn:
        return conn.execute(
            text(
                "INSERT INTO users (wa_id, role) "
                "VALUES (:wa_id, 'patient') RETURNING id"
            ),
            {"wa_id": f"91{uuid4().int % 10**10:010d}"},
        ).scalar_one()


def test_migrations_build_the_schema_the_models_describe(engine: Engine):
    with engine.connect() as conn:
        differences = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    assert differences == []


def test_downgrade_removes_everything_and_upgrade_rebuilds_it(
    engine: Engine, database_url: str, monkeypatch: pytest.MonkeyPatch
):
    migrate(database_url, monkeypatch, "base")
    assert inspect(engine).get_table_names() == ["alembic_version"]

    migrate(database_url, monkeypatch, "head")
    with engine.connect() as conn:
        assert compare_metadata(MigrationContext.configure(conn), Base.metadata) == []


def test_a_retried_whatsapp_delivery_is_not_stored_twice(engine: Engine):
    patient = add_patient(engine)
    insert = text(
        "INSERT INTO messages (wa_message_id, user_id, direction, kind, body, status) "
        "VALUES ('wamid.retried', :user_id, 'inbound', 'text', 'bukhar hai', 'pending')"
    )
    with engine.begin() as conn:
        conn.execute(insert, {"user_id": patient})

    with pytest.raises(IntegrityError), engine.begin() as conn:
        conn.execute(insert, {"user_id": patient})


def test_concurrent_workers_claim_different_pending_messages(engine: Engine):
    patient = add_patient(engine)
    with engine.begin() as conn:
        for wa_message_id in ("wamid.first", "wamid.second"):
            conn.execute(
                text(
                    "INSERT INTO messages "
                    "(wa_message_id, user_id, direction, kind, status) "
                    "VALUES (:id, :user_id, 'inbound', 'text', 'pending')"
                ),
                {"id": wa_message_id, "user_id": patient},
            )
    claim = text(
        "UPDATE messages SET status = 'processing', claimed_at = now() "
        "WHERE id = (SELECT id FROM messages WHERE status = 'pending' "
        "ORDER BY id FOR UPDATE SKIP LOCKED LIMIT 1) "
        "RETURNING wa_message_id"
    )

    with (
        engine.connect() as first,
        engine.connect() as second,
        engine.connect() as third,
    ):
        claimed_by_first = first.execute(claim).scalar_one()
        claimed_by_second = second.execute(claim).scalar_one()
        claimed_by_third = third.execute(claim).scalar_one_or_none()
        first.commit()
        second.commit()

    assert {claimed_by_first, claimed_by_second} == {"wamid.first", "wamid.second"}
    assert claimed_by_third is None


def test_nearest_chunk_is_found_by_cosine_through_the_hnsw_index(engine: Engine):
    def vector(*leading: float) -> str:
        return (
            "[" + ",".join(map(str, [*leading, *[0.0] * (1024 - len(leading))])) + "]"
        )

    with engine.begin() as conn:
        # Cosine and Euclidean distance disagree here: "far" points the same
        # way as the query but is a long way off; "near" is close but at an angle.
        for title, embedding in (("far", vector(100, 1)), ("near", vector(0.6, 0.8))):
            conn.execute(
                text(
                    "INSERT INTO kb_chunks (text, title, source_type, embedding) "
                    "VALUES ('...', :title, 'health_worker_answer', :embedding)"
                ),
                {"title": title, "embedding": embedding},
            )

    nearest = text(
        "SELECT title FROM kb_chunks "
        "ORDER BY embedding <=> CAST(:query AS vector) LIMIT 5"
    )
    with engine.begin() as conn:
        conn.execute(text("SET LOCAL enable_seqscan = off"))
        plan = "\n".join(
            conn.execute(
                text(f"EXPLAIN {nearest.text}"), {"query": vector(1)}
            ).scalars()
        )
        titles = conn.execute(nearest, {"query": vector(1)}).scalars().all()

    assert "ix_kb_chunks_embedding" in plan
    assert titles == ["far", "near"]
