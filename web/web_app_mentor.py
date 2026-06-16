# To find correct path of files
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent  # Dynamically find the root directory (one level up from the 'web' folder)
sys.path.append(str(root_dir))                     # Forcefully add the root directory to Python's system path

#### web app code: use "streamlit run .\web\web_app_mentor.py" to run it.
##########################################################
import streamlit as st
    
from src.tools.math_engine import calculate_winning_rule
from src.agents.mentor_agent import generate_mentor_message  # This agent needs API Key 

# Set up the Web Page Title
st.title("DfReman MVP: Design Rule Evaluator ⚙️")
st.write("Adjust the sliders to set your product priorities, then press 'Calculate Best Design' buttom.")

# Create the UI Sliders (Range 1 to 5, default starting at 3)
st.subheader("Set Your Priorities")
durability_slider = st.slider("How important is Durability?", min_value=1.0, max_value=5.0, value=3.0, step=0.5)
disassembly_slider = st.slider("How important is Disassembly?", min_value=1.0, max_value=5.0, value=3.0, step=0.5)
user_scores_list = [durability_slider, disassembly_slider]

# Create a Button to run the engine. This code only runs when the user clicks the button
if st.button("Calculate Best Design"):
    st.divider() # Draws a nice horizontal line

    # Run the Math Engine first
    st.write("🧮 Calculating math...") 
    math_results = calculate_winning_rule(
        user_durability=user_scores_list[0], 
        user_disassembly=user_scores_list[1]
    )

    # Display the results on the screen
    st.success(f"🏆 **Winning Rule:** {math_results['winning_text']}")
    st.info(f"**Points:** {math_results['winner_points']}")
    
    # Show the raw math points
    st.write("### The Scoreboard")
    st.write(f"**Winner rule:** {math_results['winning_text']} ({math_results['winner_points']} points)")
    st.write(f"**Loser rule:** {math_results['losing_text']} ({math_results['loser_points']} points)")

    st.divider()
    # Trigger the AI Mentor and pass it the math results
    st.write("🧠 AI Mentor is reviewing the scores and writing your report... (Please wait a few seconds)")
    
    mentor_message = generate_mentor_message(math_results, user_scores_list)  
    
    # Display the AI's message on the web page
    st.write("### Mentor's Advice")
    st.write(mentor_message)