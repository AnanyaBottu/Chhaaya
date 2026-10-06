import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import Engine, text

from chhaaya.app import app
from chhaaya.settings import Settings
from chhaaya.webhook import get_engine, get_settings

APP_SECRET = "synthetic-app-secret"
VERIFY_TOKEN = "synthetic-verify-token"


@pytest.fixture
def client(engine: Engine):
    settings = Settings(
        _env_file=None,
        postgres_password="unused",
        database_url="postgresql://unused",
        whatsapp_verify_token=VERIFY_TOKEN,
        whatsapp_app_secret=APP_SECRET,
        whatsapp_access_token="unused",
        whatsapp_phone_number_id="unused",
    )
    app.dependency_overrides[get_engine] = lambda: engine
    app.dependency_overrides[get_settings] = lambda: settings
    yield TestClient(app)
    app.dependency_overrides.clear()


def delivery(*messages: dict) -> bytes:
    """A Meta webhook body: one change that carries the given messages."""
    value = {"messaging_product": "whatsapp", "messages": list(messages)}
    change = {"field": "messages", "value": value}
    return json.dumps(
        {"object": "whatsapp_business_account", "entry": [{"changes": [change]}]}
    ).encode()


def post(client: TestClient, body: bytes, secret: str = APP_SECRET):
    signature = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return client.post(
        "/webhook",
        content=body,
        headers={
            "X-Hub-Signature-256": f"sha256={signature}",
            "Content-Type": "application/json",
        },
    )


def stored(engine: Engine) -> list[tuple]:
    with engine.connect() as conn:
        return list(
            conn.execute(
                text(
                    "SELECT m.wa_message_id, u.wa_id, m.direction, m.kind, m.body, "
                    "m.media_id, m.status FROM messages m JOIN users u "
                    "ON u.id = m.user_id ORDER BY m.id"
                )
            )
        )


def test_meta_verification_challenge_is_echoed_only_for_our_token(client: TestClient):
    ok = client.get(
        "/webhook",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": VERIFY_TOKEN,
            "hub.challenge": "1158201444",
        },
    )
    assert (ok.status_code, ok.text) == (200, "1158201444")

    wrong = client.get(
        "/webhook",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "not-ours",
            "hub.challenge": "1158201444",
        },
    )
    assert wrong.status_code == 403


def test_text_voice_and_photo_are_stored_once_even_when_meta_retries(
    client: TestClient, engine: Engine
):
    body = delivery(
        {"from": "911111111111", "id": "wamid.t", "type": "text",
         "text": {"body": "bukhar hai"}},
        {"from": "911111111111", "id": "wamid.a", "type": "audio",
         "audio": {"id": "media-audio"}},
        {"from": "912222222222", "id": "wamid.i", "type": "image",
         "image": {"id": "media-image", "caption": "report"}},
        {"from": "912222222222", "id": "wamid.s", "type": "sticker",
         "sticker": {"id": "media-sticker"}},
    )  # fmt: skip

    assert post(client, body).status_code == 200
    assert post(client, body).status_code == 200  # Meta retried the delivery

    assert stored(engine) == [
        ("wamid.t", "911111111111", "inbound", "text", "bukhar hai", None, "pending"),
        ("wamid.a", "911111111111", "inbound", "audio", None, "media-audio", "pending"),
        ("wamid.i", "912222222222", "inbound", "image", "report", "media-image",
         "pending"),
    ]  # fmt: skip
    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM users")).scalar_one() == 2


def test_status_only_deliveries_are_acknowledged_without_storing(
    client: TestClient, engine: Engine
):
    value = {"statuses": [{"id": "wamid.out", "status": "delivered"}]}
    body = json.dumps(
        {
            "object": "whatsapp_business_account",
            "entry": [{"changes": [{"field": "messages", "value": value}]}],
        }
    ).encode()

    assert post(client, body).status_code == 200
    assert stored(engine) == []


def test_missing_or_wrong_signature_is_rejected_and_nothing_is_stored(
    client: TestClient, engine: Engine
):
    body = delivery(
        {
            "from": "911111111111",
            "id": "wamid.x",
            "type": "text",
            "text": {"body": "hi"},
        }
    )

    assert client.post("/webhook", content=body).status_code == 403
    assert post(client, body, secret="someone-elses-secret").status_code == 403
    tampered = body.replace(b'"hi"', b'"bye"')
    signature = hmac.new(APP_SECRET.encode(), body, hashlib.sha256).hexdigest()
    forged = client.post(
        "/webhook",
        content=tampered,
        headers={"X-Hub-Signature-256": f"sha256={signature}"},
    )
    assert forged.status_code == 403
    garbage = client.post(
        "/webhook", content=body, headers={b"X-Hub-Signature-256": "sha256=é".encode()}
    )
    assert garbage.status_code == 403
    assert stored(engine) == []


@pytest.mark.parametrize("app_secret", [None, ""])
def test_a_blank_app_secret_is_refused_at_startup(
    monkeypatch: pytest.MonkeyPatch, tmp_path, app_secret: str | None
):
    # An empty HMAC key is public knowledge, so signatures would prove nothing.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("POSTGRES_PASSWORD", "unused")
    monkeypatch.setenv("DATABASE_URL", "postgresql://unused")
    monkeypatch.setenv("WHATSAPP_VERIFY_TOKEN", VERIFY_TOKEN)
    monkeypatch.setenv("WHATSAPP_ACCESS_TOKEN", "unused")
    monkeypatch.setenv("WHATSAPP_PHONE_NUMBER_ID", "unused")
    if app_secret is None:
        monkeypatch.delenv("WHATSAPP_APP_SECRET", raising=False)
    else:
        monkeypatch.setenv("WHATSAPP_APP_SECRET", app_secret)
    get_settings.cache_clear()
    try:
        with pytest.raises(ValidationError), TestClient(app):
            pass
    finally:
        get_settings.cache_clear()
