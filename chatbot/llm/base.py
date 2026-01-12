# chatbot/llm/base.py
from abc import ABC, abstractmethod

class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        pass
