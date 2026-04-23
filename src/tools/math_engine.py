# To find correct path of files
import sys
from pathlib import Path
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
####################################

import json

def calculate_winning_rule(user_durability, user_disassembly):
    """Calculates the winner using a weighted sum model."""
    # 1. Load the new data structure
    with open('src/tools/conflicts.json', 'r') as file:
        data = json.load(file)
    
    conflict = data["conflict_1"]
    rule_a_data = conflict["scores"]["rule_A"]
    rule_b_data = conflict["scores"]["rule_B"]
    
    # 2. The Advanced Math: (User Weight * Rule Score)
    total_a = (user_durability * rule_a_data["durability"]) + (user_disassembly * rule_a_data["disassembly"])
    total_b = (user_durability * rule_b_data["durability"]) + (user_disassembly * rule_b_data["disassembly"])
    
    # 3. Find the highest total
    if total_a > total_b:
        winner = "rule_A"
        loser = "rule_B"
        winner_points = total_a
        loser_points = total_b
    elif total_b > total_a:
        winner = "rule_B"
        loser = "rule_A"
        winner_points = total_b
        loser_points = total_a
    else:
        winner = "rule_A" # Default to A in a tie
        loser = "rule_B"
        winner_points = total_a
        loser_points = total_b
        
    # 4. Return the detailed results
    return {
        #"winning_rule": winner,
        "winning_text": conflict[f"{winner}_text"],
        #"losing_rule": loser,
        "losing_text": conflict[f"{loser}_text"],
        "winner_points": winner_points,
        "loser_points": loser_points,
    }

# 5. Test the engine
if __name__ == "__main__":
    # Pretend the user cares equally about both (Durability=3, Disassembly=3)
    print("Testing with User Durability=3, User Disassembly=3...")
    result = calculate_winning_rule(user_durability=3, user_disassembly=3)
    
    print(f"\nThe winner is: {result['winning_text']}")
    print(f"The loser is: {result['losing_text']}")
    print(f"winner points: {result['winner_points']}")
    print(f"loser points: {result['loser_points']}")
