import json
import threading
import time
from collections.abc import Callable

import httpx2
import pytest
from sqlalchemy import Engine, text

from chhaaya.sarvam import SarvamClient
from chhaaya.whatsapp import WhatsAppClient
from chhaaya.worker import Inbound, Worker


class World:
    """The outside world: Meta and Sarvam answer from here, and agents record calls."""

    def __init__(self, engine: Engine):
        self.engine = engine
        self.sent: list[tuple[str, str]] = []
        self.language: tuple[str, str] | None = ("hi-IN", "Latn")
        self.sarvam_calls = 0
        self.whatsapp_up = True
        self.agent_calls: list[tuple[str, int]] = []
        http = httpx2.Client(transport=httpx2.MockTransport(self.handle))
        self.agents: dict[str, Callable[[Inbound], None]] = {
            name: self.agent(name) for name in ("qa", "document", "escalation")
        }
        self.worker = Worker(
            engine,
            WhatsAppClient("token", "123456", http),
            SarvamClient("key", http),
            self.agents,
        )

    def agent(self, name: str) -> Callable[[Inbound], None]:
        return lambda message: self.agent_calls.append((name, message.id))

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        if request.url.host == "api.sarvam.ai":
            self.sarvam_calls += 1
            if self.language is None:
                return httpx2.Response(500, json={"error": {"message": "down"}})
            language, script = self.language
            return httpx2.Response(
                200, json={"language_code": language, "script_code": script}
            )
        if not self.whatsapp_up:
            return httpx2.Response(401, json={"error": {"message": "token expired"}})
        body = json.loads(request.read())
        self.sent.append((body["to"], body["text"]["body"]))
        return httpx2.Response(200, json={"messages": [{"id": "wamid.out"}]})

    def add_user(self, wa_id: str, role: str = "patient", asha: int | None = None):
        with self.engine.begin() as conn:
            return conn.execute(
                text(
                    "INSERT INTO users (wa_id, role, asha_id) "
                    "VALUES (:wa_id, :role, :asha) RETURNING id"
                ),
                {"wa_id": wa_id, "role": role, "asha": asha},
            ).scalar_one()

    def receive(
        self, user: int, kind: str = "text", body: str | None = None, wamid: str = ""
    ) -> int:
        with self.engine.begin() as conn:
            return conn.execute(
                text(
                    "INSERT INTO messages (wa_message_id, user_id, direction, kind, "
                    "body, status) VALUES (:wamid, :user, 'inbound', :kind, :body, "
                    "'pending') RETURNING id"
                ),
                {
                    "wamid": wamid or f"wamid.{time.monotonic_ns()}",
                    "user": user,
                    "kind": kind,
                    "body": body,
                },
            ).scalar_one()

    def message(self, message_id: int):
        with self.engine.connect() as conn:
            return conn.execute(
                text("SELECT status, error FROM messages WHERE id = :id"),
                {"id": message_id},
            ).one()

    def user(self, user_id: int):
        with self.engine.connect() as conn:
            return conn.execute(
                text("SELECT language, script FROM users WHERE id = :id"),
                {"id": user_id},
            ).one()

    def cases(self):
        with self.engine.connect() as conn:
            return conn.execute(
                text("SELECT cause, patient_id, asha_id, status FROM cases")
            ).all()


@pytest.fixture
def world(engine: Engine) -> World:
    return World(engine)


def test_two_workers_never_take_the_same_message(world: World):
    patient = world.add_user("911111111111")
    ids = [world.receive(patient, body=f"sawal number {n}") for n in range(40)]
    seen: list[int] = []

    def slow_qa(message: Inbound) -> None:
        time.sleep(0.005)
        seen.append(message.id)

    world.agents["qa"] = slow_qa

    def drain() -> None:
        while world.worker.work_one():
            pass

    threads = [threading.Thread(target=drain) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(seen) == ids
    assert all(world.message(i).status == "done" for i in ids)


def test_a_danger_sign_gets_an_urgent_reply_and_a_case_even_if_sarvam_is_down(
    world: World,
):
    world.language = None  # Sarvam is down
    asha = world.add_user("919999999999", "asha")
    patient = world.add_user("911111111111", asha=asha)
    message = world.receive(patient, body="pregnancy mein khoon aa raha hai")

    assert world.worker.work_one()

    [(to, reply)] = world.sent
    assert to == "911111111111"
    assert "108" in reply
    assert world.cases() == [("danger_sign", patient, asha, "open")]
    assert world.message(message).status == "done"
    assert world.sarvam_calls == 0
    assert world.agent_calls == []


def test_the_asha_case_opens_even_when_the_urgent_reply_cannot_be_sent(
    world: World,
):
    world.whatsapp_up = False
    asha = world.add_user("919999999999", "asha")
    patient = world.add_user("911111111111", asha=asha)
    message = world.receive(patient, body="usko jhatke aa rahe hain")

    world.worker.work_one()

    assert world.cases() == [("danger_sign", patient, asha, "open")]
    status, error = world.message(message)
    assert status == "failed"
    assert "401" in error


def test_a_danger_sign_from_a_patient_without_an_asha_is_answered_then_failed(
    world: World,
):
    patient = world.add_user("911111111111")
    message = world.receive(patient, body="My baby is not feeding since morning")

    world.worker.work_one()

    assert len(world.sent) == 1
    assert world.cases() == []
    status, error = world.message(message)
    assert status == "failed"
    assert "ASHA" in error


def test_typed_text_stores_the_language_and_goes_to_the_qa_agent(world: World):
    patient = world.add_user("911111111111")
    message = world.receive(patient, body="bukhar mein kya khana chahiye")

    world.worker.work_one()

    assert world.user(patient) == ("hi", "Latn")
    assert world.agent_calls == [("qa", message)]
    assert world.message(message).status == "done"
    assert world.sent == []


def test_an_ashas_message_skips_screening_and_a_photo_goes_to_the_document_agent(
    world: World,
):
    asha = world.add_user("919999999999", "asha")
    patient = world.add_user("911111111111", asha=asha)
    reply = world.receive(asha, body="12 usko khoon aa raha tha ab theek hai")
    photo = world.receive(patient, kind="image")

    world.worker.work_one()
    world.worker.work_one()

    assert world.agent_calls == [("escalation", reply), ("document", photo)]
    assert world.sent == []
    assert world.cases() == []
    assert world.sarvam_calls == 0


def test_a_failed_message_keeps_its_error_and_is_not_retried(world: World):
    patient = world.add_user("911111111111")
    message = world.receive(patient, body="sugar kitni honi chahiye")

    def broken(_: Inbound) -> None:
        raise RuntimeError("retrieval is down")

    world.agents["qa"] = broken

    assert world.worker.work_one()
    status, error = world.message(message)
    assert status == "failed"
    assert "RuntimeError: retrieval is down" in error
    assert not world.worker.work_one()


@pytest.mark.parametrize(
    ("kind", "body", "language", "expected"),
    [
        ("audio", None, ("hi-IN", "Deva"), "voice"),
        ("text", "vanakkam", ("ta-IN", "Taml"), "ta-IN"),
        ("text", "hello", None, "api.sarvam.ai"),
    ],
)
def test_messages_that_cannot_be_handled_fail_loudly(
    world: World, kind, body, language, expected
):
    world.language = language
    patient = world.add_user("911111111111")
    message = world.receive(patient, kind=kind, body=body)

    world.worker.work_one()

    status, error = world.message(message)
    assert status == "failed"
    assert expected in error
    assert world.agent_calls == []


def test_a_missing_agent_fails_the_message_instead_of_dropping_it(world: World):
    del world.agents["qa"]
    patient = world.add_user("911111111111")
    message = world.receive(patient, body="bukhar kyun aata hai")

    world.worker.work_one()

    status, error = world.message(message)
    assert status == "failed"
    assert "qa" in error


def test_nothing_pending_means_nothing_to_do(world: World):
    assert not world.worker.work_one()
