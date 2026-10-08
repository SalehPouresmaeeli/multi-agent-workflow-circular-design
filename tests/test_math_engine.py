### Tests for src/tools/math_engine.py — the weighted-sum design-rule scorer.
### Expected values are worked out by hand from src/tools/conflicts.json:
###   rule_A (glue):   durability 5, disassembly 1
###   rule_B (screws): durability 2, disassembly 5
###   total = user durability x rule durability + user disassembly x rule disassembly

import json
from pathlib import Path
import pytest
from src.tools.math_engine import calculate_winning_rule

CONFLICTS_PATH = Path(__file__).resolve().parent.parent / "src" / "tools" / "conflicts.json"

@pytest.fixture
def conflict():
    """The sample conflict the engine reads, so tests can compare against its rule texts."""
    with open(CONFLICTS_PATH, "r") as f:
        return json.load(f)["conflict_1"]

@pytest.mark.parametrize(
    "durability, disassembly, expected_winner, winner_points, loser_points",
    [
        (5, 1, "rule_A", 26, 15),      # A: 25 + 1,  B: 10 + 5   -> durability-heavy user picks glue
        (1, 5, "rule_B", 27, 10),      # A: 5 + 5,   B: 2 + 25   -> disassembly-heavy user picks screws
        (3, 3, "rule_B", 21, 18),      # A: 15 + 3,  B: 6 + 15   -> equal priorities still favour screws
        (2.5, 4.0, "rule_B", 25.0, 16.5),  # half-step slider values from the web app
    ],
)
def test_winner_and_points(conflict, durability, disassembly, expected_winner, winner_points, loser_points):
    expected_loser = "rule_B" if expected_winner == "rule_A" else "rule_A"

    result = calculate_winning_rule(user_durability=durability, user_disassembly=disassembly)

    assert result["winning_text"] == conflict[f"{expected_winner}_text"]
    assert result["losing_text"] == conflict[f"{expected_loser}_text"]
    assert result["winner_points"] == winner_points
    assert result["loser_points"] == loser_points

def test_tie_goes_to_rule_a(conflict):
    # durability 4, disassembly 3 -> A: 20 + 3 = 23, B: 8 + 15 = 23
    result = calculate_winning_rule(user_durability=4, user_disassembly=3)

    assert result["winner_points"] == result["loser_points"] == 23
    assert result["winning_text"] == conflict["rule_A_text"]

def test_result_has_the_fields_the_web_app_and_mentor_use():
    result = calculate_winning_rule(user_durability=3, user_disassembly=3)

    assert set(result) == {"winning_text", "losing_text", "winner_points", "loser_points"}
