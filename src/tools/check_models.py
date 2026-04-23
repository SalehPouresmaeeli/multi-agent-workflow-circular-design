from dotenv import load_dotenv
from google import genai

# This line reads .env file and loads the keys into the system
load_dotenv()
# Connect to the API (it automatically finds your system variable)
client = genai.Client()

# Loop through and print the name of every available model
for model in client.models.list():
    print(model.name)