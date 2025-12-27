import os
import requests
import json
from dotenv import load_dotenv

# Load .env file
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set (dotenv not loaded?)")

URL = (
    "https://generativelanguage.googleapis.com/"
    "v1beta/models"
    f"?key={API_KEY}"
)

resp = requests.get(URL, timeout=30)
print("HTTP status:", resp.status_code)
print(json.dumps(resp.json(), indent=2))
