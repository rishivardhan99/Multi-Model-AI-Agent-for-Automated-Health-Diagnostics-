# model3/gemini_client.py
import os
import time
import json
import random
import requests
from typing import Tuple

API_KEY = os.getenv("GEMINI_API_KEY")
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")
TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", "30"))

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in environment (.env or env var)")

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

def _endpoint_for_model(model_name: str) -> str:
    # Accept both "models/xxx" and "xxx"
    if model_name.startswith("models/"):
        model_name = model_name[len("models/"):]
    return f"{BASE_URL}/models/{model_name}:generateContent?key={API_KEY}"

def call_gemini(prompt_text: str, model_name: str = None, max_retries: int = 4) -> Tuple[int, dict, str]:
    """
    Minimal, compatible call to Gemini using the 'contents' shape.
    Returns (http_status, parsed_json_response, raw_text)
    Retries on 429/5xx with exponential backoff + jitter.
    """
    model_name = model_name or DEFAULT_MODEL
    url = _endpoint_for_model(model_name)
    headers = {"Content-Type": "application/json"}

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt_text}]
            }
        ]
    }

    backoff = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT)
        except requests.RequestException as exc:
            if attempt == max_retries:
                return 0, {"error": str(exc)}, ""
            time.sleep(backoff + random.random() * 0.5)
            backoff *= 2
            continue

        # try parse
        try:
            content = r.json()
        except Exception:
            content = {"raw_text": r.text}

        raw_text = ""
        if r.status_code == 200:
            try:
                cand = content.get("candidates", [])
                if cand and isinstance(cand, list):
                    raw_text = cand[0]["content"]["parts"][0]["text"]
                else:
                    raw_text = content.get("output", "") or content.get("result", "") or r.text
            except Exception:
                raw_text = r.text
            return r.status_code, content, raw_text

        # retriable
        if r.status_code in (429, 500, 502, 503, 504):
            if attempt == max_retries:
                return r.status_code, content, r.text
            time.sleep(backoff + random.random() * 0.5)
            backoff *= 2
            continue

        # non-retriable
        return r.status_code, content, r.text

    return 0, {"error": "max_retries_exceeded"}, ""
