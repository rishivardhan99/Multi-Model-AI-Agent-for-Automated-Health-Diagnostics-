# model3/client.py
import os
import time
import json
import random
import requests
from typing import Tuple, Optional

# Environment-driven defaults
DEFAULT_BACKEND = os.getenv("LLM_BACKEND", "gemini").lower()  # 'gemini' or 'grok'
DEFAULT_MODEL = os.getenv("LLM_MODEL", os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash"))
TIMEOUT = int(os.getenv("LLM_TIMEOUT", "30"))

# --- Gemini (Google Generative Language API) config (optional) ---
_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"

# --- Grok (example flexible endpoint) config (optional) ---
# For Grok (or any similar provider), set GROK_API_URL and GROK_API_KEY in env.
_GROK_API_URL = os.getenv("GROK_API_URL")    # e.g., "https://api.grok.example/v1/generate"
_GROK_API_KEY = os.getenv("GROK_API_KEY")

# Utility: decide backend from input
def _choose_backend(model_name: Optional[str], backend_hint: Optional[str]) -> str:
    if backend_hint:
        return backend_hint.lower()
    if model_name:
        mn = model_name.lower()
        if "grok" in mn or "xai" in mn:
            return "grok"
        if mn.startswith("models/") or "gemini" in mn:
            return "gemini"
    return DEFAULT_BACKEND

# Build Gemini endpoint
def _gemini_endpoint_for_model(model_name: str) -> str:
    if model_name.startswith("models/"):
        model_name = model_name[len("models/"):]
    return f"{_GEMINI_BASE}/models/{model_name}:generateContent?key={_GEMINI_API_KEY}"

# Internal callers
def _call_gemini(prompt_text: str, model_name: str, timeout: int) -> Tuple[int, dict, str]:
    if not _GEMINI_API_KEY:
        return 0, {"error": "GEMINI_API_KEY environment variable not set"}, ""
    url = _gemini_endpoint_for_model(model_name)
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt_text}]
            }
        ]
    }
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
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

def _call_grok(prompt_text: str, model_name: str, timeout: int) -> Tuple[int, dict, str]:
    """
    Generic Grok-style caller. The exact payload shape varies by provider.
    This implementation assumes a simple POST to GROK_API_URL with Authorization header.
    Set GROK_API_URL and GROK_API_KEY in env to use.
    """
    if not _GROK_API_URL or not _GROK_API_KEY:
        return 0, {"error": "GROK_API_URL or GROK_API_KEY not set"}, ""
    headers = {"Authorization": f"Bearer {_GROK_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model_name or "grok-1",
        "prompt": prompt_text,
        "max_tokens": 1024
    }
    r = requests.post(_GROK_API_URL, headers=headers, json=payload, timeout=timeout)
    try:
        content = r.json()
    except Exception:
        content = {"raw_text": r.text}
    raw_text = ""
    # Try a few common shapes providers use:
    if r.status_code == 200:
        # Example shapes:
        # { "text": "..." } or { "choices": [{"text":"..."}] } or { "output": "..." }
        if isinstance(content, dict):
            if "text" in content:
                raw_text = content["text"]
            elif "output" in content:
                raw_text = content["output"]
            elif "choices" in content and isinstance(content["choices"], list) and content["choices"]:
                raw_text = content["choices"][0].get("text", "")
            else:
                # Fallback to stringifying body
                raw_text = json.dumps(content)
        else:
            raw_text = str(content)
    return r.status_code, content, raw_text

# Public function
def call_llm(prompt_text: str, model_name: Optional[str] = None, backend: Optional[str] = None,
             timeout: Optional[int] = None, max_retries: int = 4) -> Tuple[int, dict, str]:
    """
    Generic LLM caller supporting multiple backends.

    Returns (http_status, parsed_json_response_or_error, raw_text)
    - model_name: provider-specific model identifier
    - backend: explicit backend string ('gemini', 'grok') or None (auto-detect)
    - max_retries: retry on retriable HTTP statuses
    """
    model_name = model_name or DEFAULT_MODEL
    timeout = timeout or TIMEOUT
    chosen = _choose_backend(model_name, backend)

    backoff = 1.0
    last_status, last_content, last_text = 0, {}, ""
    for attempt in range(1, max_retries + 1):
        try:
            if chosen == "grok":
                status, content, raw_text = _call_grok(prompt_text, model_name, timeout)
            else:
                status, content, raw_text = _call_gemini(prompt_text, model_name, timeout)
        except requests.RequestException as exc:
            last_status, last_content, last_text = 0, {"error": str(exc)}, ""
            if attempt == max_retries:
                return last_status, last_content, last_text
            time.sleep(backoff + random.random() * 0.5)
            backoff *= 2
            continue
        last_status, last_content, last_text = status, content, raw_text

        # success
        if status == 200:
            return status, content, raw_text

        # retriable codes
        if status in (429, 500, 502, 503, 504):
            if attempt == max_retries:
                return status, content, raw_text
            # attempt to extract server-suggested retry time if present
            retry_after = backoff
            try:
                # provider-specific: some return {"error": {"details":[{"retryDelay":"5s"}]}}
                retry_after = int(str(last_content.get("error", {}).get("details", [{}])[-1].get("retryDelay", f"{int(backoff)}s")).replace("s",""))
            except Exception:
                pass
            time.sleep(retry_after + random.random() * 0.5)
            backoff *= 2
            continue

        # non-retriable
        return status, content, raw_text

    return last_status, last_content, last_text
