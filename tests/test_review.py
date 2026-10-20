import pytest
from sqlalchemy import Engine
from test_cases import setup
from test_worker import World

from chhaaya import review
from chhaaya.retrieval import KnowledgeBase


@pytest.fixture
def world(engine: Engine) -> World:
    world = World(engine)
    world.agents["escalation"] = world.cases_.answer
    return world


def answer_a_case(world: World, answer: str) -> None:
    asha, patient = setup(world)
    number = world.cases_.open("weak_evidence", patient, "kya ORS roz de sakte hain")
    world.receive(asha, body=f"{number} {answer}")
    world.worker.work_one()


def test_an_asha_answer_is_searchable_only_after_a_person_approves_it(world: World):
    answer_a_case(world, "Haan, dast mein ORS roz dein")
    kb = KnowledgeBase(world.engine, world.embed)

    assert kb.search("ORS") == []
    [(chunk_id, text)] = review.pending(world.engine)
    assert text == "Haan, dast mein ORS roz dein"

    review.approve(world.engine, chunk_id)

    assert [hit.text for hit in kb.search("ORS")] == ["Haan, dast mein ORS roz dein"]
    assert review.pending(world.engine) == []


def test_a_rejected_answer_is_never_searchable_and_cannot_be_approved(world: World):
    answer_a_case(world, "Sunita ko ORS dein, uska ghar school ke paas hai")
    [(chunk_id, _)] = review.pending(world.engine)

    review.reject(world.engine, chunk_id)

    assert KnowledgeBase(world.engine, world.embed).search("ORS") == []
    assert review.pending(world.engine) == []
    with pytest.raises(LookupError):
        review.approve(world.engine, chunk_id)
