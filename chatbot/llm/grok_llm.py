# chatbot/llm/grok_llm.py
import os
import requests
from .base import BaseLLM

class GrokLLM(BaseLLM):
    def __init__(self, api_url: str = None, api_key: str = None, model: str = None, timeout: int = 15):
        self.api_url = api_url or os.getenv("GROK_API_URL")
        self.api_key = api_key or os.getenv("GROK_API_KEY")
        self.model = model or os.getenv("MEDICUBE_LLM_MODEL")
        self.timeout = timeout

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        if not self.api_url or not self.api_key:
            return "[LLM not configured] Grok API key or URL missing."
        # Example HTTP contract — adapt to real API if available
        try:
            payload = {"prompt": prompt, "model": self.model, "max_tokens": max_tokens}
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            resp = requests.post(self.api_url, headers=headers, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            # adapt to returned shape
            if isinstance(data, dict):
                if "text" in data:
                    return data["text"]
                if "choices" in data and data["choices"]:
                    return data["choices"][0].get("text", str(data))
            return str(data)
        except Exception as e:
            return f"[LLM error] {e}"
