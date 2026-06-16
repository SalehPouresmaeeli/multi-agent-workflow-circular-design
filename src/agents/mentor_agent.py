# To find correct path of files
import sys
from pathlib import Path
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
####################################

from crewai import Agent, Task, Crew
from src.tools.llm_config import get_llm
# This agent needs API Key
def generate_mentor_message(math_results, user_scores_list):

    my_llm = get_llm("mentor_agent")

    # Create the Mentor Agent
    mentor = Agent(
        role="Design Mentor",
        goal="Explain engineering trade-offs to the user in a friendly, easy-to-understand way.",
        backstory="You are an experienced, patient engineering mentor. You help users understand why a specific design rule won based on their priorities.",
        llm=my_llm
    )

    # Give the Agent its Task (Injecting the exact math results)
    explain_task = Task(
        description=f"""
        User priorities are '{user_scores_list[0]}' for Durability and '{user_scores_list[1]} for Disassembly'.
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

    # Assemble the Crew and Run
    crew = Crew(
        agents=[mentor],
        tasks=[explain_task]
    )

    # Return the text result back to the Streamlit UI
    result = crew.kickoff()
    return result.raw
