import os
import json
import requests

# Optional: load .env if you used it
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in environment")

URL = (
    "https://generativelanguage.googleapis.com/"
    "v1beta/models/gemini-2.5-flash:generateContent"
    f"?key={API_KEY}"
)


# Minimal, deterministic-safe prompt
payload = {
    "contents": [
        {
            "role": "user",
            "parts": [
                {
                    "text": (
                        "Return ONLY valid JSON.\n"
                        "Explain this lab finding conservatively.\n\n"
                        "Facts:\n"
                        "- LDL: 142 mg/dL\n"
                        "- HDL: 38 mg/dL\n"
                        "- Status: borderline high\n\n"
                        "JSON format:\n"
                        "{ \"summary\": \"...\" }"
                    )
                }
            ]
        }
    ]
}

response = requests.post(
    URL,
    headers={"Content-Type": "application/json"},
    json=payload,
    timeout=30
)

print("HTTP status:", response.status_code)

data = response.json()
print(json.dumps(data, indent=2))

# Extract text safely
try:
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    print("\n=== MODEL OUTPUT ===")
    print(text)
except Exception as e:
    print("Could not extract text:", e)
