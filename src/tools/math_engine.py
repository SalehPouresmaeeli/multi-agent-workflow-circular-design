import json

def calculate_winning_rule(durability_score, disassembly_score):
    """Calculates the winning rule based on user slider scores."""
    # 1. Open and read our dummy data
    with open('conflicts.json', 'r') as file:
        data = json.load(file)
    
    conflict = data["conflict_1"]
    
    # 2. Start the scoreboard at 0
    scores = {"rule_A": 0, "rule_B": 0}
    
    # 3. Add the user's scores to the matching attributes
    scores[conflict["attributes"]["durability"]] += durability_score
    scores[conflict["attributes"]["disassembly"]] += disassembly_score
    
    # 4. Find the highest score
    if scores["rule_A"] > scores["rule_B"]:
        winner = "rule_A"
    elif scores["rule_B"] > scores["rule_A"]:
        winner = "rule_B"
    else:
        winner = "rule_A" # Default to A in a tie
        
    # 5. Return a clean dictionary with all the data we need for the next steps
    return {
        "winning_rule": winner,
        "winning_text": conflict[winner],
        "final_scores": scores
    }

# 6. Test the engine locally
if __name__ == "__main__":
    print("Testing with Durability=5, Disassembly=2...")
    result = calculate_winning_rule(durability_score=5, disassembly_score=2)
    
    print(f"\nThe winner is: {result['winning_rule']}")
    print(f"Rule text: '{result['winning_text']}'")
    print(f"Scoreboard: {result['final_scores']}")
    