# chatbot/llm/gemini_llm.py

import os
import time
import random
import requests
from typing import Tuple
from chatbot.llm.base import BaseLLM

# ---- Environment (same contract as model3) ----
API_KEY = os.getenv("GEMINI_API_KEY")
DEFAULT_MODEL = os.getenv("MEDICUBE_LLM_MODEL", os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash"))
TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", "30"))

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in environment (.env or env var)")

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


def _endpoint_for_model(model_name: str) -> str:
    """
    Accept both 'models/xxx' and 'xxx'
    """
    if model_name.startswith("models/"):
        model_name = model_name[len("models/"):]
    return f"{BASE_URL}/models/{model_name}:generateContent?key={API_KEY}"


class GeminiLLM(BaseLLM):
    """
    Gemini adapter for chatbot.
    Uses the SAME request + retry logic as model3,
    but implemented locally (no imports, no SDK).
    """

    def __init__(self, model: str = None, max_retries: int = 4):
        self.model = model or DEFAULT_MODEL
        self.max_retries = max_retries

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """
        Generate response using Gemini HTTP API.
        Returns raw text output or raises RuntimeError on failure.
        """

        url = _endpoint_for_model(self.model)
        headers = {"Content-Type": "application/json"}

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ]
        }

        backoff = 1.0

        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=TIMEOUT
                )
            except requests.RequestException as exc:
                if attempt == self.max_retries:
                    raise RuntimeError(f"Gemini request failed: {exc}")
                time.sleep(backoff + random.random() * 0.5)
                backoff *= 2
                continue

            # Try parsing JSON
            try:
                data = response.json()
            except Exception:
                data = {"raw_text": response.text}

            # Success
            if response.status_code == 200:
                try:
                    candidates = data.get("candidates", [])
                    if candidates and isinstance(candidates, list):
                        return candidates[0]["content"]["parts"][0]["text"].strip()
                    # fallback fields
                    return (
                        data.get("output")
                        or data.get("result")
                        or response.text
                    ).strip()
                except Exception:
                    return response.text.strip()

            # Retriable errors
            if response.status_code in (429, 500, 502, 503, 504):
                if attempt == self.max_retries:
                    raise RuntimeError(
                        f"Gemini failed after retries "
                        f"(status={response.status_code}): {data}"
                    )
                time.sleep(backoff + random.random() * 0.5)
                backoff *= 2
                continue

            # Non-retriable error
            raise RuntimeError(
                f"Gemini non-retriable error "
                f"(status={response.status_code}): {data}"
            )

        raise RuntimeError("Gemini max retries exceeded")
