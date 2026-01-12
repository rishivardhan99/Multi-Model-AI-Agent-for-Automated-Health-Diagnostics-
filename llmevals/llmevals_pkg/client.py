# llmevals/llmevals_pkg/client.py
import os
import json
import time
import random
from typing import Dict, Any, Optional
import requests


class GroqRateLimitError(RuntimeError):
    pass


def _backoff_sleep(attempt: int, base: float = 0.5):
    jitter = random.uniform(0, 0.5)
    time.sleep(base * (2 ** attempt) + jitter)


class LLMClient:
    """Simple LLM client with retries and support for:
    - groq (any client_type that startswith 'groq') -> POST JSON to GROQ_API_URL with Authorization Bearer GROQ_API_KEY
    - openai_compat (any client_type that startswith 'openai') -> POST to OPENAI_API_BASE /completions with OPENAI_API_KEY
    """

    def __init__(self, cfg: dict):
        self.client_type = (cfg.get('CLIENT_TYPE') or os.getenv('CLIENT_TYPE') or 'groq_http')
        self.model = cfg.get('MODEL_NAME') or os.getenv('MODEL_NAME')
        # GROQ settings
        self.groq_key = cfg.get('GROQ_API_KEY') or os.getenv('GROQ_API_KEY')
        self.groq_url = cfg.get('GROQ_API_URL') or os.getenv('GROQ_API_URL') or 'https://api.groq.ai/v1/models/{model}/outputs'
        # OPENAI-compat
        self.openai_key = cfg.get('OPENAI_API_KEY') or os.getenv('OPENAI_API_KEY')
        self.openai_base = cfg.get('OPENAI_API_BASE') or os.getenv('OPENAI_API_BASE')

        if self.client_type.lower().startswith('groq') and not self.groq_key:
            raise RuntimeError('GROQ_API_KEY is not set for groq client type')
        if self.client_type.lower().startswith('openai') and not self.openai_key:
            raise RuntimeError('OPENAI_API_KEY is not set for openai client type')

    def _estimate_chars_from_tokens(self, tokens: int) -> int:
        # conservative char estimate: 1 token ~= 4 chars (approx)
        return int(tokens * 4)

    def call(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.0, timeout: int = 90) -> Dict[str, Any]:
        t = self.client_type.lower()
        if t.startswith('groq'):
            return self._call_groq(prompt, max_tokens, temperature, timeout)
        elif t.startswith('openai'):
            return self._call_openai_compat(prompt, max_tokens, temperature, timeout)
        else:
            raise RuntimeError(f'Unsupported client type: {self.client_type}')

    def _call_groq(self, prompt: str, max_tokens: int, temperature: float, timeout: int) -> Dict[str, Any]:
        url = self.groq_url  # should be https://api.groq.com/openai/v1/chat/completions

        headers = {
            'Authorization': f'Bearer {self.groq_key}',
            'Content-Type': 'application/json',
        }

        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a strict JSON-only evaluation assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }

        last_err = None
        for attempt in range(6):
            try:
                resp = requests.post(url, headers=headers, json=body, timeout=timeout)
                j = resp.json()
            except Exception as e:
                last_err = e
                _backoff_sleep(attempt)
                continue

            if resp.status_code == 200:
                try:
                    return {
                        "text": j["choices"][0]["message"]["content"],
                        "raw": j
                    }
                except Exception:
                    return {"text": json.dumps(j), "raw": j}

            if resp.status_code == 429:
                _backoff_sleep(attempt)
                continue

            raise RuntimeError(f"Groq API error {resp.status_code}: {j}")

        raise RuntimeError(f"Groq call failed after retries: {last_err}")

    def _call_openai_compat(self, prompt: str, max_tokens: int, temperature: float, timeout: int) -> Dict[str, Any]:
        base = self.openai_base or 'https://api.openai.com/v1'
        url = f'{base}/completions'
        headers = {
            'Authorization': f'Bearer {self.openai_key}',
            'Content-Type': 'application/json',
        }
        body = {
            'model': self.model,
            'prompt': prompt,
            'temperature': float(temperature),
            'max_tokens': int(max_tokens),
            'n': 1,
        }

        last_err: Optional[Exception] = None
        for attempt in range(6):
            try:
                resp = requests.post(url, headers=headers, json=body, timeout=timeout)
            except requests.exceptions.RequestException as e:
                last_err = e
                _backoff_sleep(attempt)
                continue
            try:
                j = resp.json()
            except Exception:
                resp.raise_for_status()
            if resp.status_code == 200:
                if 'choices' in j and j['choices']:
                    return {'text': j['choices'][0].get('text', ''), 'raw': j}
                return {'text': json.dumps(j), 'raw': j}
            if resp.status_code in (429, 413):
                if resp.status_code == 413:
                    raise RuntimeError(f'OpenAI compat API error 413: {j}')
                retry_after = None
                try:
                    retry_after = int(resp.headers.get('Retry-After')) if resp.headers.get('Retry-After') else None
                except Exception:
                    retry_after = None
                if retry_after:
                    time.sleep(retry_after + 0.5)
                    continue
                last_err = RuntimeError(f'OpenAI compat API error 429: {j}')
                _backoff_sleep(attempt)
                continue
            raise RuntimeError(f'OpenAI compat API error {resp.status_code}: {j}')

        raise RuntimeError(f'OpenAI call failed after retries: {last_err}')
