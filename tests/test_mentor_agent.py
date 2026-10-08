### Tests for src/agents/mentor_agent.py — building the Design Mentor's task from the math engine's results.
### The CrewAI crew is replaced, so the prompt can be checked without calling an LLM.

from src.agents import mentor_agent
from fakes import fake_crew_class

MATH_RESULTS = {
    "winning_text": "Use standard modular screws to assemble the outer casing.",
    "losing_text": "Use heavy-duty industrial glue to assemble the outer casing.",
    "winner_points": 25.0,
    "loser_points": 16.5,
}


def test_mentor_message_is_returned(monkeypatch):
    FakeCrew, _ = fake_crew_class(raw_result="Screws won because you prioritised disassembly.")
    monkeypatch.setattr(mentor_agent, "Crew", FakeCrew)

    message = mentor_agent.generate_mentor_message(MATH_RESULTS, [2.5, 4.0])

    assert message == "Screws won because you prioritised disassembly."


def test_task_contains_priorities_rules_and_points(monkeypatch):
    FakeCrew, crews = fake_crew_class()
    monkeypatch.setattr(mentor_agent, "Crew", FakeCrew)

    mentor_agent.generate_mentor_message(MATH_RESULTS, [2.5, 4.0])

    [crew] = crews
    description = crew.tasks[0].description
    assert "'2.5' for Durability" in description
    assert "4.0" in description
    assert MATH_RESULTS["winning_text"] in description
    assert MATH_RESULTS["losing_text"] in description
    assert "Winner Points: 25.0" in description
    assert "Loser Points: 16.5" in description
    assert crew.agents[0].role == "Design Mentor"
