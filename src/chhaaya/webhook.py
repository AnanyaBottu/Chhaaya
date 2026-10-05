import hashlib
import hmac
import json
import logging
from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import PlainTextResponse
from sqlalchemy import Engine, create_engine
from sqlalchemy.dialects.postgresql import insert

from chhaaya.db import Message, User, sqlalchemy_url
from chhaaya.settings import Settings

log = logging.getLogger("chhaaya.webhook")
router = APIRouter()

KINDS = ("text", "audio", "image")


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_engine() -> Engine:
    return create_engine(sqlalchemy_url(get_settings().database_url))


@router.get("/webhook", response_class=PlainTextResponse)
def verify(
    mode: Annotated[str, Query(alias="hub.mode")],
    token: Annotated[str, Query(alias="hub.verify_token")],
    challenge: Annotated[str, Query(alias="hub.challenge")],
    settings: Annotated[Settings, Depends(get_settings)],
) -> str:
    if mode != "subscribe" or not hmac.compare_digest(
        token, settings.whatsapp_verify_token
    ):
        raise HTTPException(403, "verification failed")
    return challenge


@router.post("/webhook")
async def receive(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    engine: Annotated[Engine, Depends(get_engine)],
) -> dict[str, str]:
    body = await request.body()
    expected = hmac.new(
        settings.whatsapp_app_secret.encode(), body, hashlib.sha256
    ).hexdigest()
    signature = request.headers.get("X-Hub-Signature-256", "")
    if not hmac.compare_digest(signature, f"sha256={expected}"):
        raise HTTPException(403, "invalid signature")
    await run_in_threadpool(store, engine, json.loads(body))
    return {"status": "ok"}


def store(engine: Engine, payload: dict) -> None:
    """Queue every text, voice and photo message; repeated deliveries are no-ops."""
    values = (
        change.get("value", {})
        for entry in payload.get("entry", [])
        for change in entry.get("changes", [])
    )
    messages = [m for value in values for m in value.get("messages", [])]
    with engine.begin() as conn:
        for message in messages:
            kind = message["type"]
            if kind not in KINDS:
                log.warning("ignored %s message %s", kind, message["id"])
                continue
            # The no-op update makes RETURNING give back the id of an existing user.
            user_id = conn.execute(
                insert(User)
                .values(wa_id=message["from"], role="patient")
                .on_conflict_do_update(
                    index_elements=["wa_id"], set_={"wa_id": message["from"]}
                )
                .returning(User.id)
            ).scalar_one()
            content = message[kind]
            conn.execute(
                insert(Message)
                .values(
                    wa_message_id=message["id"],
                    user_id=user_id,
                    direction="inbound",
                    kind=kind,
                    body=content.get("body") or content.get("caption"),
                    media_id=content.get("id"),
                    status="pending",
                )
                .on_conflict_do_nothing(index_elements=["wa_message_id"])
            )
