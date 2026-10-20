import subprocess
import sys
from pathlib import Path

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


@pytest.mark.parametrize("chunk_id", ["999", "abc"])
def test_a_wrong_id_fails_with_one_line_and_no_traceback(
    world: World, tmp_path: Path, chunk_id: str
):
    answer_a_case(world, "Haan, dast mein ORS roz dein")
    command = [sys.executable, "-m", "chhaaya.review", "approve", chunk_id]

    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)

    assert result.returncode == 1
    assert len(result.stderr.splitlines()) == 1, result.stderr
    assert [text for _, text in review.pending(world.engine)] == [
        "Haan, dast mein ORS roz dein"
    ]
