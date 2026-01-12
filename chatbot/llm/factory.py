# chatbot/llm/factory.py
import os
from chatbot.llm.base import BaseLLM

# import adapters
from chatbot.llm.gemini_llm import GeminiLLM
from chatbot.llm.grok_llm import GrokLLM

class StubLLM(BaseLLM):
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        return "[LLM not configured]"

def get_llm() -> BaseLLM:
    provider = os.getenv("MEDICUBE_LLM_PROVIDER", "gemini").lower()
    model = os.getenv("MEDICUBE_LLM_MODEL", None)
    try:
        if provider == "gemini":
            return GeminiLLM(model=model)
        if provider == "grok":
            return GrokLLM(model=model)
    except Exception:
        return StubLLM()
    return StubLLM()
