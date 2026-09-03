import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# Request information about the specific key
response = requests.get(
    "https://openrouter.ai/api/v1/key",
    headers={"Authorization": f"Bearer {api_key}"}
)

key_data = response.json()
print(key_data)