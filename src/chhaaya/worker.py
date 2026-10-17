"""The worker: claims stored messages and turns them into replies (design section 3)."""

import logging
import re
import signal
import threading
from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy import Engine, create_engine, text

from chhaaya import danger_signs
from chhaaya.cases import open_case
from chhaaya.db import sqlalchemy_url
from chhaaya.sarvam import SarvamClient
from chhaaya.settings import Settings
from chhaaya.whatsapp import WhatsAppClient

log = logging.getLogger("chhaaya.worker")
DEVANAGARI = re.compile("[\u0900-\u097f]")

# Take the oldest pending message. SKIP LOCKED makes a second worker pass over
# a message the first is claiming, so no message is processed twice.
CLAIM = text("""
    WITH claimed AS (
        UPDATE messages SET status = 'processing', claimed_at = now()
        WHERE id = (
            SELECT id FROM messages WHERE status = 'pending'
            ORDER BY id FOR UPDATE SKIP LOCKED LIMIT 1
        )
        RETURNING id, user_id, kind, body, media_id
    )
    SELECT c.id, c.user_id, c.kind, c.body, c.media_id,
           u.wa_id, u.role, u.asha_id, u.language, u.script
    FROM claimed c JOIN users u ON u.id = c.user_id
""")


@dataclass(frozen=True)
class Inbound:
    id: int
    user_id: int
    kind: str
    body: str | None
    media_id: str | None
    wa_id: str
    role: str
    asha_id: int | None
    language: str | None
    script: str | None


@dataclass
class Worker:
    engine: Engine
    whatsapp: WhatsAppClient
    sarvam: SarvamClient
    # "qa", "document" and "escalation" each take the message and reply.
    agents: dict[str, Callable[[Inbound], None]]

    def work_one(self) -> bool:
        """Process the oldest pending message; False when there is none."""
        with self.engine.begin() as conn:
            row = conn.execute(CLAIM).one_or_none()
        if row is None:
            return False
        message = Inbound(**row._mapping)
        try:
            self.process(message)
        except Exception as error:
            log.exception("message %s failed", message.id)
            self.finish(message.id, "failed", f"{type(error).__name__}: {error}")
        else:
            self.finish(message.id, "done")
        return True

    def finish(self, message_id: int, status: str, error: str | None = None) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE messages SET status = :status, error = :error "
                    "WHERE id = :id"
                ),
                {"status": status, "error": error, "id": message_id},
            )

    def process(self, message: Inbound) -> None:
        if message.role == "asha":
            return self.route("escalation", message)
        if message.kind == "audio":
            raise NotImplementedError("voice notes are not supported yet")
        if message.body and (signs := danger_signs.detect(message.body)):
            return self.escalate(message, signs)
        if message.kind == "text":
            self.store_language(message)
        self.route("document" if message.kind == "image" else "qa", message)

    def route(self, agent: str, message: Inbound) -> None:
        if agent not in self.agents:
            raise LookupError(f"no {agent} agent is registered")
        self.agents[agent](message)

    def store_language(self, message: Inbound) -> None:
        language, script = self.sarvam.identify_language(message.body or "")
        with self.engine.begin() as conn:
            conn.execute(
                text("UPDATE users SET language = :l, script = :s WHERE id = :id"),
                {"l": language, "s": script, "id": message.user_id},
            )

    def escalate(self, message: Inbound, signs: list[str]) -> None:
        """Reply first, and open the ASHA's case even if the reply fails."""
        log.warning("danger signs %s in message %s", signs, message.id)
        try:
            self.whatsapp.send_text(message.wa_id, urgent_reply(message))
        finally:
            if message.asha_id is not None:
                with self.engine.begin() as conn:
                    open_case(conn, "danger_sign", message.user_id, message.asha_id)
        if message.asha_id is None:
            raise LookupError("patient has no ASHA to alert")


def urgent_reply(message: Inbound) -> str:
    """The urgent reply in the patient's language. Reading it must not need Sarvam."""
    replies = danger_signs.URGENT_REPLY
    if message.language == "en":
        return replies["en"]
    if message.language == "hi":
        return replies["hi" if message.script == "Deva" else "hi-Latn"]
    if DEVANAGARI.search(message.body or ""):
        return replies["hi"]
    # Latin script from someone we know nothing about: English or Hinglish.
    return f"{replies['en']}\n\n{replies['hi-Latn']}"


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = Settings()
    worker = Worker(
        create_engine(sqlalchemy_url(settings.database_url)),
        WhatsAppClient(
            settings.whatsapp_access_token, settings.whatsapp_phone_number_id
        ),
        SarvamClient(settings.sarvam_api_key),
        agents={},  # the Q&A, document and escalation agents register here as they land
    )
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    log.info("worker started")
    while not stop.is_set():
        if not worker.work_one():
            stop.wait(1)
    log.info("worker stopped")


if __name__ == "__main__":
    main()
