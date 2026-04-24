import sys
from pathlib import Path

# Fix for Windows console UnicodeEncodeError
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from crewai import Agent, Task, Crew, LLM
    
# 1. Import our custom math calculator from the other file
from src.tools.math_engine import calculate_winning_rule

# 2. Run the math engine to get the facts (simulating user sliders at 3 and 3)
user_score = {
    "durability": 3,
    "disassembly": 3
}
math_results = calculate_winning_rule(user_score["durability"], user_score["disassembly"])

# 3. Set up LLM
my_llm = LLM(model="gemini/gemini-3.1-flash-lite")

# 4. Create the Mentor Agent
mentor = Agent(
    role="Design Mentor",
    goal="Explain engineering trade-offs to the user in a friendly, easy-to-understand way.",
    backstory="You are an experienced, patient engineering mentor. You help users understand why a specific design rule won based on their priorities.",
    llm=my_llm
)

# 5. Give the Agent its Task (Injecting the exact math results)
explain_task = Task(
    description=f"""
    User priorities are '{user_score}'
    The math engine has calculated the winning design rule. Here is the data:
    - Winning rule: '{math_results['winning_text']}'
    - Losing rule: '{math_results['losing_text']}'
    - Winner Points: {math_results['winner_points']}
    - Loser Points: {math_results['loser_points']}

    Write a short, technical message to the user. Tell them which rule won and explain that it won because of user specific priorities. Keep it to 3 sentences max. Also, mention Winner Points and Loser Points.
    """,
    expected_output="A short, 3-sentence technical explanation of the results.",
    agent=mentor
)

# 6. Assemble the Crew and Run
crew = Crew(
    agents=[mentor],
    tasks=[explain_task]
)

print("The Mentor is reviewing the math and writing a message...\n")
result = crew.kickoff()
print("\n--- MENTOR'S MESSAGE ---")
print(result)
